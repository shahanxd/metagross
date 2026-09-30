"""OPTIONAL research reference: frozen DINOv2-S features + linear head on OFFROAD5.

Purpose: one "how good can a strong frozen foundation-model feature get with only a
linear head" row next to the deployable LR-ASPP. Not deployed (ViT-S/14 is too slow for
the live loop on the target CPU). Two backbones:

* DINOv2 ViT-S/14 (Meta, Apache-2.0) via ``timm`` (``vit_small_patch14_dinov2.lvd142m``).
* The same backbone with FiT3D weights (Yue et al., "Improving 2D Feature
  Representations by 3D-Aware Fine-Tuning", ECCV 2024), loaded with ``--fit3d-ckpt``.
  Download the DINOv2-small FiT3D checkpoint from the authors' release
  (https://github.com/ywyue/FiT3D) first; keys are matched non-strictly and the number
  of missing/unexpected keys is logged and stored in the JSON so a bad load is visible.

Protocol (DINOv2 linear evaluation, single layer): patch tokens of the last block
(after the final norm) -> BatchNorm -> 1x1 conv to 5 classes -> bilinear upsampling
to the label resolution. Backbone frozen (no grad), CE with ENet class weights, AdamW.
Evaluation uses the project metrics (mIoU, false-safe rate, per-source split).

    python aws/teacher_dinov2.py --data-root data/offroad5 --out results/seg_teacher_dinov2s.json
    python aws/teacher_dinov2.py --data-root data/offroad5 --fit3d-ckpt ckpts/fit3d_dinov2_small.pth \
        --out results/seg_teacher_fit3d_dinov2s.json

``--backbone stub`` swaps in a tiny strided conv so the loop can be unit-tested on CPU.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metagross.eval.seg_eval import SegAccumulator  # noqa: E402
from metagross.train.datasets import SegDataset, build_splits, class_frequencies, enet_class_weights  # noqa: E402
from metagross.train.train_seg import parse_hw, torch_confusion  # noqa: E402

LOG = logging.getLogger("teacher_dinov2")
PATCH = 14
TIMM_NAME = "vit_small_patch14_dinov2.lvd142m"
NUM_CLASSES = 5
_PREFIXES = ("module.", "backbone.", "model.", "encoder.")


class TimmDinoBackbone(nn.Module):
    """Frozen timm DINOv2 ViT returning (B, C, H/14, W/14) patch features."""

    def __init__(self, fit3d_ckpt: Optional[Path] = None) -> None:
        super().__init__()
        import timm

        self.m = timm.create_model(TIMM_NAME, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.load_report: dict[str, Any] = {"weights": TIMM_NAME}
        if fit3d_ckpt is not None:
            sd = torch.load(fit3d_ckpt, map_location="cpu", weights_only=False)
            for key in ("model", "state_dict", "teacher"):
                if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
            clean = {}
            for k, v in sd.items():
                for p in _PREFIXES:
                    if k.startswith(p):
                        k = k[len(p):]
                clean[k] = v
            res = self.m.load_state_dict(clean, strict=False)
            n_model = len(self.m.state_dict())
            self.load_report = {"weights": f"FiT3D {fit3d_ckpt}", "missing_keys": len(res.missing_keys), "unexpected_keys": len(res.unexpected_keys), "model_keys": n_model}
            LOG.info("FiT3D load: %s", self.load_report)
            if len(res.missing_keys) > n_model // 2:
                raise RuntimeError(f"FiT3D checkpoint does not match {TIMM_NAME}: {self.load_report}")
        self.embed_dim = int(self.m.embed_dim)
        self.n_prefix = int(getattr(self.m, "num_prefix_tokens", 1))
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.m.forward_features(x)[:, self.n_prefix :]
        b, n, c = tok.shape
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        return tok.transpose(1, 2).reshape(b, c, h, w)


class StubBackbone(nn.Module):
    """Test double: a fixed random strided conv with the same output geometry."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, PATCH, stride=PATCH)
        self.embed_dim = dim
        self.load_report = {"weights": "stub"}
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def linear_head(dim: int) -> nn.Module:
    return nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, NUM_CLASSES, 1))


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    hw = parse_hw(args.img)
    if hw[0] % PATCH or hw[1] % PATCH:
        raise ValueError(f"--img must be a multiple of {PATCH}")
    train_s, val_s = build_splits("offroad5", args.data_root, args.train_subset, args.val_subset, args.seed)
    cls_w = torch.from_numpy(enet_class_weights(class_frequencies(train_s, seed=args.seed))).to(dev)
    train_ds, val_ds = SegDataset(train_s, hw, "clean", seed=args.seed), SegDataset(val_s, hw, "none")
    backbone = (StubBackbone() if args.backbone == "stub" else TimmDinoBackbone(args.fit3d_ckpt)).to(dev).eval()
    head = linear_head(backbone.embed_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = dev.type == "cuda"
    total = args.epochs * max(1, len(train_ds) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))
    history = []
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        train_ds.set_epoch(ep)
        g = torch.Generator().manual_seed(args.seed * 7919 + ep)
        head.train()
        for x, y, _ in DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, generator=g, num_workers=args.workers, pin_memory=amp):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                feats = backbone(x)
            logits = F.interpolate(head(feats.float()), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y, weight=cls_w, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        acc = SegAccumulator()
        with torch.no_grad():
            for x, y, idx in DataLoader(val_ds, batch_size=args.bs, num_workers=args.workers):
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    feats = backbone(x)
                pred = F.interpolate(head(feats.float()), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
                for b in range(x.shape[0]):
                    acc.cms.append(torch_confusion(y[b], pred[b]).cpu().numpy())
                    acc.sources.append(val_ds.samples[int(idx[b])].source)
                    acc.lumas.append(float("nan"))
        rep = acc.report()
        rep.pop("brightness", None)
        history.append({"epoch": ep, "loss": float(loss.detach()), "val": rep})
        LOG.info("epoch %d loss %.4f val mIoU %.4f false-safe %s", ep, float(loss.detach()), rep["overall"]["miou"] or 0, rep["overall"]["false_safe_rate"])
    best = max(history, key=lambda r: r["val"]["overall"]["miou"] or 0)
    out = {
        "schema": "metagross.seg_teacher/1",
        "claim_label": "Tested",
        "note": "Research reference only (frozen ViT-S/14 + linear head); not deployable on the target CPU.",
        "backbone": backbone.load_report,
        "img_hw": list(hw),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_train": len(train_s),
        "n_val": len(val_s),
        "history": history,
        "best_epoch": best["epoch"],
        "best_val": best["val"],
        "wall_s": round(time.perf_counter() - t0, 1),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    LOG.info("wrote %s", args.out)
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DINOv2-S / FiT3D frozen-feature linear probe on OFFROAD5.")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "offroad5")
    ap.add_argument("--backbone", choices=("dinov2_s", "stub"), default="dinov2_s")
    ap.add_argument("--fit3d-ckpt", type=Path, default=None)
    ap.add_argument("--img", default="322x420", help="HxW, multiples of 14")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--val-subset", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
