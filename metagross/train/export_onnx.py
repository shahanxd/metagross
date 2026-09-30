"""Export a trained checkpoint to ONNX (opset 17, static 1x3xHxW) and verify it.

The exported graph maps an ImageNet-normalised RGB tensor (1, 3, H, W) float32 to
five-class logits (1, 5, H, W) at the input resolution. Verification runs the ONNX
Runtime CPU session against the PyTorch model on a random tensor and on real
validation images and records the maximum absolute logit difference and the argmax
agreement. A sidecar ``<name>.json`` (read by
:class:`metagross.autonomy.perception.semantics.Segmenter`) records classes, colours,
normalisation, input size, training data, validation metrics and licence.

Usage::

    python -m metagross.train.export_onnx --ckpt runs/seg/lraspp_smoke/best.pt \
        --out models/lraspp_smoke.onnx --data-root data/rugd5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from metagross.autonomy.perception.semantics import SEM_COLORS
from metagross.contracts.interfaces import SEM_CLASSES
from metagross.train.models import build_model

LOG = logging.getLogger("export_onnx")

OPSET = 17
MAX_ABS_DIFF_FAIL = 1e-2  # logits; float32 CPU exports typically agree to ~1e-4
N_VERIFY_IMAGES = 4

LICENCE = {
    "code": "torchvision LR-ASPP MobileNetV3-Large architecture (BSD-3-Clause)",
    "backbone_init": "torchvision ImageNet-1k MobileNetV3-Large weights (BSD-3-Clause distribution; ImageNet data terms apply)",
    "training_data": "GAIA-URJC RUGD-5Labels-resized / OFFROAD5 (RUGD, RELLIS-3D, GOOSE) mirrors: CC BY-NC-SA 3.0",
    "weights_use": "research / evaluation only (derived from non-commercial data). A production BEL deployment "
    "must retrain with the same pipeline on commercially licensed or own-collected data.",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(ckpt_path: Path) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Rebuild the model from a ``best.pt``/``last.pt``; returns (model.eval(), meta, ckpt)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = build_model(meta["args"]["model"], meta["num_classes"], init="none")
    model.load_state_dict(ck["model"])
    return model.eval(), meta, ck


def _verify_inputs(meta: dict[str, Any], data_root: Optional[Path], seed: int = 0) -> list[np.ndarray]:
    """One random normal tensor plus up to N_VERIFY_IMAGES real val images (normalised)."""
    h, w = meta["img_hw"]
    xs = [np.random.default_rng(seed).standard_normal((1, 3, h, w)).astype(np.float32)]
    if data_root is not None and (data_root / "val").exists():
        from metagross.train.augment import resize_pair
        from metagross.train.data_index import list_samples, read_mask, read_rgb, subset
        from metagross.train.datasets import normalize_image

        for s in subset(list_samples(data_root, "val"), N_VERIFY_IMAGES, seed):
            img, _ = resize_pair(read_rgb(s.image), read_mask(s.mask), (h, w))
            xs.append(normalize_image(img)[None])
    return xs


def export(ckpt_path: Path, out_path: Path, data_root: Optional[Path] = None, name: Optional[str] = None) -> dict[str, Any]:
    """Export + verify + write sidecar. Returns the sidecar dictionary."""
    import onnx
    import onnxruntime as ort

    model, meta, ck = load_checkpoint(ckpt_path)
    h, w = meta["img_hw"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, h, w)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,  # TorchScript exporter: no onnxscript dependency, static shapes
    )
    onnx.checker.check_model(onnx.load(str(out_path)))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])
    max_diff, agree, n_px = 0.0, 0, 0
    with torch.no_grad():
        for x in _verify_inputs(meta, data_root):
            ref = model(torch.from_numpy(x)).numpy()
            got = sess.run(None, {"input": x})[0]
            max_diff = max(max_diff, float(np.abs(ref - got).max()))
            agree += int((ref.argmax(1) == got.argmax(1)).sum())
            n_px += ref.shape[0] * ref.shape[2] * ref.shape[3]
    agreement = agree / max(n_px, 1)
    LOG.info("ORT vs torch: max |dlogit| = %.3g, argmax agreement = %.6f", max_diff, agreement)
    if max_diff > MAX_ABS_DIFF_FAIL:
        raise RuntimeError(f"ONNX export mismatch: max abs diff {max_diff:.3g} > {MAX_ABS_DIFF_FAIL}")

    args = meta["args"]
    sidecar = {
        "name": name or out_path.stem,
        "kind": "direct5",
        "architecture": f"{args['model']} (torchvision lraspp_mobilenet_v3_large)" if args["model"] == "lraspp" else args["model"],
        "params": meta.get("params"),
        "input_hw": [h, w],
        "input": "input: (1, 3, H, W) float32, RGB, (x/255 - mean) / std",
        "output": "logits: (1, 5, H, W) float32",
        "mean": meta["mean"],
        "std": meta["std"],
        "classes": {str(k): v for k, v in SEM_CLASSES.items()},
        "colors_rgb": {str(k): list(v) for k, v in SEM_COLORS.items()},
        "smoke": bool(meta.get("smoke")) or "smoke" in out_path.stem,
        "training": {
            "data": args["data"],
            "aug": args["aug"],
            "init": args["init"],
            "img_hw": meta["img_hw"],
            "batch_size": args["bs"],
            "lr": args["lr"],
            "sched": args["sched"],
            "epochs_planned": args["epochs"],
            "best_epoch": ck.get("epoch"),
            "iterations_at_best": ck.get("global_iter"),
            "iterations_planned": meta.get("total_iters"),
            "partial": bool(meta.get("total_iters")) and (ck.get("global_iter") or 0) < meta["total_iters"],
            "sampling": args.get("sampling", "uniform"),
            "repeat_factor_sampling": meta.get("repeat_factor_sampling"),
            "n_train_images": meta["n_train"],
            "n_val_images": meta["n_val"],
            "train_list_sha": meta["train_list_sha"],
            "class_weights": meta["class_weights"],
            "device": meta["device"],
            "torch": meta["torch"],
        },
        "val_metrics_at_train_res": {k: ck.get("val", {}).get(k) for k in ("miou", "pixel_acc", "false_safe_rate", "per_class_iou")},
        "licence": LICENCE,
        "export": {
            "opset": OPSET,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "max_abs_logit_diff_vs_torch": max_diff,
            "argmax_agreement_vs_torch": agreement,
            "sha256": sha256_file(out_path),
            "source_checkpoint": ckpt_path.as_posix(),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    LOG.info("wrote %s (+ sidecar)", out_path)
    return sidecar


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export a METAGROSS segmentation checkpoint to ONNX.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="models/<name>.onnx")
    ap.add_argument("--data-root", type=Path, default=None, help="val images for verification")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(2)
    export(args.ckpt, args.out, args.data_root, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
