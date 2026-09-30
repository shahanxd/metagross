"""Train the five-class terrain segmenter (LR-ASPP MobileNetV3 by default).

Recipe: ImageNet-pretrained backbone, class-weighted CE + 0.5 x Dice, AdamW with linear
warm-up then poly (or cosine) decay, per-epoch validation mIoU / false-safe rate, best
checkpoint by val mIoU, full resume support, CUDA + AMP when available.

Examples::

    # AWS GPU (see aws/setup_and_train.sh)
    python -m metagross.train.train_seg --model lraspp --data offroad5 --aug robust \
        --epochs 40 --bs 32 --lr 6e-4 --img 320x416 --amp --workers 8 --out runs/seg/offroad5_robust

    # laptop pipeline check (tiny subset, a few CPU iterations)
    python -m metagross.train.train_seg --smoke --out runs/seg/smoke_tiny

    # budgeted CPU run: 250-step "epochs" (val + checkpoint each), repeat-factor sampling,
    # exact mid-epoch resume, hard wall-clock stop (see scripts/finish_seg_cpu.ps1)
    python -m metagross.train.train_seg --data rugd5 --aug robust --img 256x320 --bs 8 \
        --threads 2 --workers 1 --epochs 12 --epoch-iters 250 --sampling repeat \
        --val-subset 300 --deadline 09:25 --resume --out runs/seg_cpu

Outputs in ``--out``: ``last.pt`` (full state, for --resume), ``best.pt`` (weights + meta),
``metrics.json`` (args, per-epoch history, best), ``train.log``. Checkpoints are written
atomically (temp file + rename) so a concurrent reader never sees a half-written file.

``--epoch-iters N`` (> 0) switches to the step-indexed stream sampler of
:mod:`metagross.train.sampling`: an epoch is N optimiser steps, and ``--epochs`` counts
these steps-epochs. ``--epoch-iters 0`` (default) keeps one shuffled pass per epoch.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from metagross.autonomy.perception.semantics import IMAGENET_MEAN, IMAGENET_STD, NUM_SEM_CLASSES
from metagross.eval.seg_eval import SegAccumulator
from metagross.train.data_index import list_fingerprint
from metagross.train.datasets import DATASETS, SegDataset, build_splits, class_frequencies, enet_class_weights
from metagross.train.losses import CEDiceLoss
from metagross.train.models import INITS, MODELS, build_model, count_params
from metagross.train.sampling import (
    RFS_MIN_FRAC_DEFAULT,
    RFS_THRESH_DEFAULT,
    StreamDataset,
    build_stream,
    epoch_step_range,
    image_class_fractions,
    repeat_factors,
    stream_length,
)

LOG = logging.getLogger("train_seg")
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_EVERY_ITERS = 20
GRAD_CLIP_NORM = 5.0
MIN_LR_RATIO = 0.01  # final lr = MIN_LR_RATIO * base lr
POLY_POWER = 0.9
SAVE_RETRIES = 10  # os.replace can fail on Windows while another process reads the target
SAVE_RETRY_S = 1.0  # seconds between retries

SMOKE_DEFAULTS = {"bs": 4, "epochs": 1, "img": "128x160", "train_subset": 32, "val_subset": 8, "max_iters": 6, "workers": 0, "threads": 2}
FULL_DEFAULTS = {"bs": 16, "epochs": 40, "img": "320x416", "train_subset": 0, "val_subset": 0, "max_iters": 0, "workers": 4, "threads": 0}


def parse_hw(s: str) -> tuple[int, int]:
    """'320x416' -> (320, 416) = (height, width)."""
    h, w = s.lower().split("x")
    return int(h), int(w)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the METAGROSS terrain segmenter.")
    ap.add_argument("--model", choices=MODELS, default="lraspp")
    ap.add_argument("--data", choices=DATASETS, default="rugd5")
    ap.add_argument("--data-root", type=Path, default=None, help="default: <repo>/data/<data>")
    ap.add_argument("--aug", choices=("clean", "robust"), default="clean")
    ap.add_argument("--init", choices=INITS, default="imagenet")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--sched", choices=("poly", "cosine"), default="poly")
    ap.add_argument("--warmup-iters", type=int, default=None, help="default: min(500, 5%% of total)")
    ap.add_argument("--img", default=None, help="HxW training crop, e.g. 320x416")
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA (bf16 if supported else fp16)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-subset", type=int, default=None, help="seeded subset of training images (0 = all)")
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None, help="stop after this many optimiser steps (0 = no cap)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to prove the pipeline")
    ap.add_argument("--epoch-iters", type=int, default=0,
                    help="optimiser steps per epoch (val + checkpoint period); > 0 uses the step-indexed stream sampler "
                         "(exact mid-epoch resume). 0 = one shuffled pass per epoch")
    ap.add_argument("--sampling", choices=("uniform", "repeat"), default="uniform",
                    help="repeat = image-level repeat-factor sampling of rare classes (implies the stream sampler)")
    ap.add_argument("--rfs-thresh", type=float, default=RFS_THRESH_DEFAULT, help="RFS threshold t (image fraction)")
    ap.add_argument("--rfs-min-frac", type=float, default=RFS_MIN_FRAC_DEFAULT, help="pixel fraction for a class to count as present")
    ap.add_argument("--deadline", default=None,
                    help="local wall-clock stop, 'HH:MM' (today) or 'YYYY-MM-DDTHH:MM': the current step finishes, then "
                         "validation + checkpoint, then exit (resumable)")
    return ap


def parse_deadline(s: Optional[str], now: Optional[_dt.datetime] = None) -> Optional[float]:
    """'HH:MM' (today, local) or ISO 'YYYY-MM-DDTHH:MM[:SS]' -> POSIX timestamp (s); None passes through."""
    if not s:
        return None
    now = now or _dt.datetime.now()
    if "T" in s or "-" in s:
        dt = _dt.datetime.fromisoformat(s)
    else:
        hh, mm = s.split(":")
        dt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    return dt.timestamp()


def atomic_save(save_fn: Any, obj: Any, path: Path, retries: int = SAVE_RETRIES, wait_s: float = SAVE_RETRY_S) -> bool:
    """``save_fn(obj, tmp)`` then rename onto ``path``; retries while the target is locked.

    Returns False (and leaves ``<path>.tmp``) if the rename never succeeded.
    """
    tmp = path.with_name(path.name + ".tmp")
    save_fn(obj, tmp)
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            LOG.warning("rename %s -> %s blocked (attempt %d/%d)", tmp.name, path.name, attempt + 1, retries)
            time.sleep(wait_s)
    return False


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset options from the smoke / full defaults."""
    defaults = SMOKE_DEFAULTS if args.smoke else FULL_DEFAULTS
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.smoke and args.device == "auto":
        args.device = "cpu"
    if args.data_root is None:
        args.data_root = REPO_ROOT / "data" / args.data
    return args


def lr_lambda_factory(total_iters: int, warmup: int, sched: str):
    """Per-iteration multiplier: linear warm-up, then poly/cosine decay to MIN_LR_RATIO."""

    def f(it: int) -> float:
        if warmup > 0 and it < warmup:
            return (it + 1) / warmup
        t = min(1.0, (it - warmup) / max(1, total_iters - warmup))
        if sched == "cosine":
            return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * t))
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * (1 - t) ** POLY_POWER

    return f


def param_groups(model: torch.nn.Module, wd: float) -> list[dict[str, Any]]:
    """No weight decay on norms and biases (1-D tensors)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def torch_confusion(gt: torch.Tensor, pred: torch.Tensor, n: int = NUM_SEM_CLASSES) -> torch.Tensor:
    """(n, n) int64 confusion on-device; gt pixels >= n (ignore) are skipped."""
    valid = gt < n
    return torch.bincount(gt[valid] * n + pred[valid], minlength=n * n).reshape(n, n)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: Optional[torch.dtype], sources: list[str]) -> dict[str, Any]:
    """Val metrics at the training resolution (squash-resize, as onboard)."""
    model.eval()
    acc = SegAccumulator()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        pred = logits.argmax(1)
        for b in range(x.shape[0]):
            cm = torch_confusion(y[b], pred[b]).cpu().numpy()
            acc.cms.append(cm)
            acc.sources.append(sources[int(idx[b])])
            acc.lumas.append(float("nan"))
    model.train()
    rep = acc.report()
    rep.pop("brightness", None)
    return rep


def save_json(path: Path, obj: Any) -> None:
    atomic_save(lambda o, p: p.write_text(json.dumps(o, indent=1), encoding="utf-8"), obj, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run training; returns the final metrics dictionary (also written to metrics.json)."""
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    if args.threads:
        torch.set_num_threads(args.threads)
    import cv2

    cv2.setNumThreads(1 if args.workers else 2)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype: Optional[torch.dtype] = None
    if args.amp and use_cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    img_hw = parse_hw(args.img)
    train_s, val_s = build_splits(args.data, args.data_root, args.train_subset, args.val_subset, args.seed)
    if not train_s or not val_s:
        raise RuntimeError(f"empty split under {args.data_root}")
    freq = class_frequencies(train_s, seed=args.seed)
    cls_w = enet_class_weights(freq)
    LOG.info("train=%d val=%d | class freq %s | weights %s", len(train_s), len(val_s), np.round(freq, 4).tolist(), np.round(cls_w, 3).tolist())

    train_ds = SegDataset(train_s, img_hw, args.aug, seed=args.seed)
    val_ds = SegDataset(val_s, img_hw, "none")
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=use_cuda)
    if len(train_ds) // args.bs == 0:
        raise RuntimeError("batch size larger than the training set")
    use_stream = args.epoch_iters > 0 or args.sampling == "repeat"
    iters_per_epoch = (args.epoch_iters or len(train_ds) // args.bs) if use_stream else len(train_ds) // args.bs
    total_iters = args.epochs * iters_per_epoch
    if args.max_iters:
        total_iters = min(total_iters, args.max_iters)
    warmup = args.warmup_iters if args.warmup_iters is not None else min(500, max(1, total_iters // 20))
    deadline_ts = parse_deadline(args.deadline)
    rfs_info: Optional[dict[str, Any]] = None
    stream_ds: Optional[StreamDataset] = None
    if use_stream:
        factors = None
        if args.sampling == "repeat":
            t_rfs = time.perf_counter()
            rf = repeat_factors(image_class_fractions(train_s), args.rfs_thresh, args.rfs_min_frac)
            factors, rfs_info = rf.image, rf.summary()
            LOG.info("repeat-factor sampling (t=%.3f, min_frac=%.4f): %s [%.1f s]", args.rfs_thresh, args.rfs_min_frac, rfs_info, time.perf_counter() - t_rfs)
        stream_ds = StreamDataset(train_ds, build_stream(len(train_ds), stream_length(total_iters, args.bs), args.seed, factors))

    model = build_model(args.model, NUM_SEM_CLASSES, args.init).to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_iters, warmup, args.sched))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    loss_fn = CEDiceLoss(torch.from_numpy(cls_w)).to(device)

    start_epoch, global_it, best_miou = 0, 0, -1.0
    history: list[dict[str, Any]] = []
    last_ckpt = out / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, global_it, best_miou, history = ck["epoch"] + 1, ck["global_iter"], ck["best_miou"], ck["history"]
        if use_stream:  # a deadline stop may have ended an epoch early: continue inside it
            start_epoch = global_it // iters_per_epoch
        LOG.info("resumed from %s at epoch %d (iter %d, best mIoU %.4f)", last_ckpt, start_epoch, global_it, best_miou)

    meta = {
        "schema": "metagross.seg_train/1",
        "smoke": bool(args.smoke),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "img_hw": list(img_hw),
        "num_classes": NUM_SEM_CLASSES,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "class_freq": freq.tolist(),
        "class_weights": cls_w.tolist(),
        "n_train": len(train_s),
        "n_val": len(val_s),
        "train_list_sha": list_fingerprint(train_s),
        "val_list_sha": list_fingerprint(val_s),
        "params": count_params(model),
        "device": str(device),
        "amp_dtype": str(amp_dtype) if amp_dtype else None,
        "torch": torch.__version__,
        "host": platform.platform(),
        "total_iters": total_iters,
        "iters_per_epoch": iters_per_epoch,
        "warmup_iters": warmup,
        "sampler": "stream" if use_stream else "epoch_shuffle",
        "repeat_factor_sampling": rfs_info,
    }
    model.train()
    t_start = time.perf_counter()
    stop = global_it >= total_iters
    stopped_by_deadline = False
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        if deadline_ts is not None and time.time() >= deadline_ts:
            LOG.warning("deadline %s passed before epoch %d: stopping", args.deadline, epoch)
            stopped_by_deadline = True
            break
        if stream_ds is not None:
            s0, s1 = epoch_step_range(epoch, iters_per_epoch, global_it, total_iters)
            if s1 <= s0:
                continue
            loader = DataLoader(
                stream_ds, batch_size=args.bs, sampler=range(s0 * args.bs, s1 * args.bs), drop_last=True,
                num_workers=args.workers, pin_memory=use_cuda, persistent_workers=False,
            )
        else:
            train_ds.set_epoch(epoch)
            gen = torch.Generator().manual_seed(args.seed * 100003 + epoch)
            loader = DataLoader(
                train_ds, batch_size=args.bs, shuffle=True, drop_last=True, generator=gen, num_workers=args.workers, pin_memory=use_cuda, persistent_workers=False
            )
        t_ep = time.perf_counter()
        run_loss, n_seen = 0.0, 0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss, parts = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_it += 1
            run_loss += float(loss.detach())
            n_seen += 1
            if global_it % LOG_EVERY_ITERS == 0 or global_it == 1:
                el = time.perf_counter() - t_ep
                LOG.info(
                    "ep %d it %d/%d loss %.4f (ce %.4f dice %.4f) lr %.2e | %.2f s/it",
                    epoch, global_it, total_iters, float(loss.detach()), parts["ce"], parts["dice"], opt.param_groups[0]["lr"], el / max(n_seen, 1),
                )
            if global_it >= total_iters:
                stop = True
                break
            if deadline_ts is not None and time.time() >= deadline_ts:
                LOG.warning("deadline %s reached at iter %d/%d: validating, checkpointing and stopping", args.deadline, global_it, total_iters)
                stop = stopped_by_deadline = True
                break
        if n_seen == 0:
            continue
        train_s_per_it = (time.perf_counter() - t_ep) / max(n_seen, 1)
        t_val = time.perf_counter()
        rep = validate(model, val_loader, device, amp_dtype, val_ds.sources)
        o = rep["overall"]
        row = {
            "epoch": epoch,
            "global_iter": global_it,
            "train_loss": run_loss / max(n_seen, 1),
            "lr": opt.param_groups[0]["lr"],
            "val_miou": o["miou"],
            "val_pixel_acc": o["pixel_acc"],
            "val_false_safe_rate": o["false_safe_rate"],
            "val_per_class_iou": o["per_class_iou"],
            "val_per_source_miou": {k: v["miou"] for k, v in rep["per_source"].items()},
            "train_s_per_iter": round(train_s_per_it, 4),
            "val_s": round(time.perf_counter() - t_val, 1),
        }
        history.append(row)
        LOG.info("epoch %d: val mIoU %.4f acc %.4f false-safe %s", epoch, o["miou"] or 0, o["pixel_acc"] or 0, o["false_safe_rate"])
        improved = (o["miou"] or 0.0) > best_miou
        if improved:
            best_miou = o["miou"] or 0.0
            atomic_save(torch.save, {"model": model.state_dict(), "meta": meta, "epoch": epoch, "global_iter": global_it, "val": o}, out / "best.pt")
        atomic_save(
            torch.save,
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_iter": global_it,
                "best_miou": best_miou,
                "history": history,
                "meta": meta,
            },
            last_ckpt,
        )
        best_row = max(history, key=lambda r: r["val_miou"] or 0.0)
        metrics = dict(meta, history=history, best=best_row, wall_s=round(time.perf_counter() - t_start, 1),
                       global_iter=global_it, complete=global_it >= total_iters, stopped_by_deadline=stopped_by_deadline,
                       updated_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        save_json(out / "metrics.json", metrics)
    logging.getLogger().removeHandler(fh)
    fh.close()
    if not (out / "metrics.json").exists():
        raise RuntimeError(f"no epoch completed in {out} (deadline already passed?)")
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = resolve_args(build_argparser().parse_args(argv))
    m = train(args)
    LOG.info("done: best val mIoU %.4f at epoch %d -> %s", m["best"]["val_miou"] or 0, m["best"]["epoch"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
