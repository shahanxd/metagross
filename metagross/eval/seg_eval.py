"""Offline evaluation of the five-class terrain segmenter.

Metrics (all computed from 5x5 confusion matrices; rows = ground truth, cols = prediction):

* per-class IoU, mIoU (mean over classes whose union is non-zero, mmseg convention),
  ``miou_present`` (mean over classes that occur in the ground truth), pixel accuracy;
* **false-safe rate** - the safety metric this project cares about most: the fraction
  of ground-truth *hazard* pixels (1 obstacle, 2 water/mud) that the model labels
  *traversable* (3 unstable, 4 stable). Reported overall and per hazard class;
* false-hazard rate - traversable pixels labelled hazard (costs availability, not safety);
* brightness-stratified mIoU: images split into terciles of mean Rec.601 luma;
* per-source breakdown (rugd / rellis / goose) for OFFROAD5;
* mean normalised entropy on correct vs wrong pixels (is the uncertainty informative?);
* latency of the full ``Segmenter.__call__`` (pre + ORT + post) at 2 and 4 threads.

CLI (writes ``results/*.json`` and deck figures)::

    python -m metagross.eval.seg_eval --model models/segformer_b0_ade.onnx \
        --data-root data/rugd5 --splits val test --out results/seg_zeroshot.json \
        --fig-prefix deck_assets/seg_zeroshot --label "zero-shot SegFormer-B0 ADE20K"

Evaluation is at the dataset label resolution: the segmenter resizes the image to the
model input, predicts, and nearest-upsamples class ids back (exactly what runs onboard).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from metagross.autonomy.perception.semantics import (
    HAZARD_CLASSES,
    NUM_SEM_CLASSES,
    SEM_PALETTE,
    TRAVERSABLE_CLASSES,
    Segmenter,
)
from metagross.contracts.interfaces import SEM_CLASSES
from metagross.train.data_index import IGNORE_INDEX, list_fingerprint, list_samples, read_mask, read_rgb, subset

LOG = logging.getLogger(__name__)

SCHEMA = "metagross.seg_eval/1"
CLASS_NAMES: tuple[str, ...] = tuple(SEM_CLASSES[i] for i in range(NUM_SEM_CLASSES))
SHORT_NAMES: tuple[str, ...] = ("sky/bg", "obstacle", "water/mud", "unstable", "stable")
BRIGHTNESS_BINS = ("dark", "mid", "bright")
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # Rec.601 luma from RGB


# --------------------------------------------------------------------------- core metrics
def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_SEM_CLASSES, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """(C, C) int64 confusion matrix, rows = GT class, cols = predicted class.

    GT pixels equal to ``ignore_index`` (or any value >= C) are skipped. Predictions must
    lie in [0, C).
    """
    g = np.asarray(gt).ravel()
    p = np.asarray(pred).ravel()
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch gt {np.shape(gt)} vs pred {np.shape(pred)}")
    valid = (g != ignore_index) & (g < num_classes)
    g = g[valid].astype(np.int64)
    p = p[valid].astype(np.int64)
    if p.size and (p.min() < 0 or p.max() >= num_classes):
        raise ValueError("prediction ids outside [0, num_classes)")
    return np.bincount(g * num_classes + p, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU = TP / (TP + FP + FN); NaN where the union is empty."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def _rate(num: float, den: float) -> Optional[float]:
    return float(num / den) if den > 0 else None


def false_safe_rate(cm: np.ndarray, hazard: Sequence[int] = HAZARD_CLASSES, traversable: Sequence[int] = TRAVERSABLE_CLASSES) -> Optional[float]:
    """Fraction of GT hazard pixels predicted traversable (None if no hazard pixels)."""
    h = list(hazard)
    t = list(traversable)
    return _rate(cm[np.ix_(h, t)].sum(), cm[h].sum())


def summarize(cm: np.ndarray) -> dict[str, Any]:
    """All scalar metrics of one confusion matrix, JSON-ready (None for undefined)."""
    cm = np.asarray(cm, dtype=np.int64)
    iou = iou_per_class(cm)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    total = int(cm.sum())

    def nanmean(x: np.ndarray) -> Optional[float]:
        x = x[~np.isnan(x)]
        return float(x.mean()) if x.size else None

    hz, tr = list(HAZARD_CLASSES), list(TRAVERSABLE_CLASSES)
    return {
        "n_pixels": total,
        "miou": nanmean(iou),
        "miou_present": nanmean(np.where(support > 0, iou, np.nan)),
        "pixel_acc": _rate(tp.sum(), total),
        "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(CLASS_NAMES, iou)},
        "per_class_recall": {n: _rate(tp[i], support[i]) for i, n in enumerate(CLASS_NAMES)},
        "per_class_precision": {n: _rate(tp[i], cm[:, i].sum()) for i, n in enumerate(CLASS_NAMES)},
        "gt_class_fraction": {n: _rate(support[i], total) for i, n in enumerate(CLASS_NAMES)},
        "false_safe_rate": false_safe_rate(cm),
        "false_safe_obstacle": _rate(cm[1, tr].sum(), support[1]),
        "false_safe_water": _rate(cm[2, tr].sum(), support[2]),
        "false_hazard_rate": _rate(cm[np.ix_(tr, hz)].sum(), cm[tr].sum()),
        "confusion": cm.tolist(),
    }


def image_luminance(rgb: np.ndarray) -> float:
    """Mean Rec.601 luma of an (H, W, 3) uint8 RGB image, in [0, 255]."""
    return float((rgb.reshape(-1, 3).astype(np.float32) @ LUMA_WEIGHTS).mean())


@dataclass
class SegAccumulator:
    """Collects one confusion matrix per image so results can be stratified afterwards."""

    num_classes: int = NUM_SEM_CLASSES
    cms: list[np.ndarray] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    lumas: list[float] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)  # scene names ("" when unknown)
    ent_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [correct, wrong]
    ent_cnt: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def add(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        source: str = "all",
        luma: float = float("nan"),
        entropy: Optional[np.ndarray] = None,
        sequence: str = "",
    ) -> np.ndarray:
        """Add one image; returns its confusion matrix."""
        cm = confusion_matrix(gt, pred, self.num_classes)
        self.cms.append(cm)
        self.sources.append(source)
        self.lumas.append(float(luma))
        self.sequences.append(sequence)
        if entropy is not None:
            valid = gt < self.num_classes
            correct = (gt == pred) & valid
            wrong = (gt != pred) & valid
            self.ent_sum += [float(entropy[correct].sum()), float(entropy[wrong].sum())]
            self.ent_cnt += [int(correct.sum()), int(wrong.sum())]
        return cm

    def total(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.cms:
            return np.zeros((self.num_classes, self.num_classes), np.int64)
        stack = np.stack(self.cms)
        return stack[mask].sum(axis=0) if mask is not None else stack.sum(axis=0)

    def report(self) -> dict[str, Any]:
        """Overall + per-source + brightness-tercile summaries."""
        out: dict[str, Any] = {"n_images": len(self.cms), "overall": summarize(self.total())}
        src = np.array(self.sources)
        out["per_source"] = {s: dict(summarize(self.total(src == s)), n_images=int((src == s).sum())) for s in sorted(set(self.sources))}
        if len(self.sequences) == len(self.cms) and any(self.sequences):
            seq = np.array(self.sequences)
            hz, tr = list(HAZARD_CLASSES), list(TRAVERSABLE_CLASSES)
            out["per_sequence"] = {}
            for name in sorted(set(self.sequences)):
                cm = self.total(seq == name)
                s = summarize(cm)
                out["per_sequence"][name] = {
                    "n_images": int((seq == name).sum()),
                    "miou": s["miou"],
                    "false_safe_rate": s["false_safe_rate"],
                    "hazard_px": int(cm[hz].sum()),
                    "false_safe_px": int(cm[np.ix_(hz, tr)].sum()),
                }
        lum = np.array(self.lumas, dtype=np.float64)
        if len(lum) >= 3 and np.isfinite(lum).all():
            t1, t2 = np.quantile(lum, [1.0 / 3.0, 2.0 / 3.0])
            bins = np.digitize(lum, [t1, t2])  # 0 dark, 1 mid, 2 bright
            strata: dict[str, Any] = {"luma_tercile_thresholds": [float(t1), float(t2)]}
            for b, name in enumerate(BRIGHTNESS_BINS):
                sel = bins == b
                s = summarize(self.total(sel))
                strata[name] = {
                    "n_images": int(sel.sum()),
                    "luma_range": [float(lum[sel].min()), float(lum[sel].max())] if sel.any() else None,
                    "miou": s["miou"],
                    "pixel_acc": s["pixel_acc"],
                    "false_safe_rate": s["false_safe_rate"],
                    "per_class_iou": s["per_class_iou"],
                }
            out["brightness"] = strata
        if self.ent_cnt.sum() > 0:
            out["entropy"] = {
                "mean_on_correct": _rate(self.ent_sum[0], self.ent_cnt[0]),
                "mean_on_wrong": _rate(self.ent_sum[1], self.ent_cnt[1]),
            }
        return out


# --------------------------------------------------------------------------- latency
def benchmark_latency(model_path: Path, image: np.ndarray, threads: Sequence[int] = (2, 4), n_iter: int = 30, warmup: int = 5) -> dict[str, Any]:
    """Wall-clock latency of ``Segmenter.__call__`` on ``image`` for each thread count (ms)."""
    res: dict[str, Any] = {"image_hw": list(image.shape[:2]), "n_iter": n_iter}
    for t in threads:
        seg = Segmenter(model_path, intra_op_threads=t)
        for _ in range(warmup):
            seg(image)
        tot, inf = [], []
        for _ in range(n_iter):
            seg(image)
            tot.append(seg.last_timings_ms["seg_total"])
            inf.append(seg.last_timings_ms["seg_infer"])
        res[f"threads_{t}"] = {
            "total_median_ms": round(statistics.median(tot), 2),
            "total_p90_ms": round(float(np.percentile(tot, 90)), 2),
            "infer_median_ms": round(statistics.median(inf), 2),
            "min_total_ms": round(min(tot), 2),
        }
        LOG.info("latency %s threads=%d: %s", model_path.name, t, res[f"threads_{t}"])
    return res


# --------------------------------------------------------------------------- evaluation
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qualitative_indices(n_samples: int, n_qualitative: int) -> list[int]:
    """Evenly spread, deterministic sample indices for the qualitative grid."""
    if not n_qualitative or not n_samples:
        return []
    return sorted(set(np.linspace(0, n_samples - 1, n_qualitative).round().astype(int).tolist()))


def evaluate_split(
    seg: Segmenter,
    data_root: Path,
    split: str,
    max_images: int = 0,
    seed: int = 0,
    n_qualitative: int = 6,
) -> tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Run ``seg`` over one split. Returns (report, qualitative samples [(rgb, gt, pred)])."""
    samples = subset(list_samples(data_root, split), max_images, seed)
    acc = SegAccumulator()
    q_idx = set(qualitative_indices(len(samples), n_qualitative))
    qual: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        rgb = read_rgb(s.image)
        gt = read_mask(s.mask)
        if gt.shape != rgb.shape[:2]:  # defensive: evaluate at label resolution
            import cv2

            rgb = cv2.resize(rgb, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        pred, ent = seg(rgb)
        acc.add(gt, pred, s.source, image_luminance(rgb), ent, sequence=s.sequence)
        if i in q_idx:
            qual.append((rgb, gt, pred))
        if (i + 1) % 250 == 0:
            LOG.info("%s: %d/%d images (%.1f s)", split, i + 1, len(samples), time.perf_counter() - t0)
    rep = acc.report()
    rep.update(
        split=split,
        data_root=str(data_root.as_posix()),
        file_list_sha=list_fingerprint(samples),
        max_images=max_images or None,
        wall_s=round(time.perf_counter() - t0, 1),
    )
    return rep, qual


# --------------------------------------------------------------------------- figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    """Row-normalised confusion matrix (each row = where that GT class went), single-hue ramp."""
    plt = _mpl()
    cm = np.asarray(cm, dtype=np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=200)
    ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    n = cm.shape[0]
    for r in range(n):
        for c in range(n):
            v = norm[r, c]
            txt = "-" if rows[r, 0] == 0 else f"{100 * v:.1f}%"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 else "#1f2328")
    ax.set_xticks(range(n), SHORT_NAMES, rotation=30, ha="right", fontsize=8.5, color="#3d434a")
    ax.set_yticks(range(n), [f"{s}\n({100 * rows[i, 0] / max(cm.sum(), 1):.1f}% px)" for i, s in enumerate(SHORT_NAMES)], fontsize=8, color="#3d434a")
    ax.set_xlabel("predicted\n(dashed box = false-safe: hazard pixels predicted traversable)", fontsize=8.5, color="#3d434a")
    ax.set_ylabel("ground truth (share of pixels)", fontsize=9, color="#3d434a")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    # outline the false-safe block: GT hazard rows (1,2) x predicted traversable cols (3,4)
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((2.5, 0.5), 2, 2, fill=False, lw=1.6, ec="#dc322f", ls="--"))
    ax.set_title(title, fontsize=9.5, color="#1f2328", loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_qualitative(samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    """Grid of image | ground truth | prediction rows with the project colour key legend."""
    plt = _mpl()
    from matplotlib.patches import Patch

    from metagross.autonomy.perception.semantics import colorize

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9.0, 2.35 * n + 0.8), dpi=150, squeeze=False)
    for r, (rgb, gt, pred) in enumerate(samples):
        for c, (im, lab) in enumerate(((rgb, "image"), (colorize(gt), "ground truth"), (colorize(pred), "prediction"))):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(lab, fontsize=10, color="#1f2328")
    handles = [Patch(facecolor=SEM_PALETTE[i] / 255.0, edgecolor="#9aa0a6", label=f"{i} {SHORT_NAMES[i]}") for i in range(NUM_SEM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=NUM_SEM_CLASSES, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=10.5, color="#1f2328", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- CLI
def _claims(label: str, tag: str, out_path: Path, splits: dict[str, Any], latency: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows ready for results/claims.csv (the ledger owner registers them)."""
    rows = []
    for sp, rep in splits.items():
        o = rep["overall"]
        for key in ("miou", "pixel_acc", "false_safe_rate"):
            if o.get(key) is not None:
                rows.append({"id": f"{tag}_{sp}_{key}", "value": round(o[key], 4), "label": "Tested", "source": out_path.as_posix(), "note": label})
    for k, v in latency.items():
        if k.startswith("threads_"):
            rows.append({"id": f"{tag}_latency_{k}_ms", "value": v["total_median_ms"], "label": "Tested", "source": out_path.as_posix(), "note": "Segmenter.__call__ median on this laptop"})
    return rows


def _measure_latency(args: argparse.Namespace) -> dict[str, Any]:
    """Latency on a dataset image and on the same image resized to the camera frame size."""
    import cv2

    from metagross.config import defaults

    ref = read_rgb(list_samples(args.data_root, args.splits[0])[0].image)
    cam = cv2.resize(ref, (defaults.IMG_W, defaults.IMG_H), interpolation=cv2.INTER_LINEAR)
    return {
        "dataset_image": benchmark_latency(args.model, ref, args.latency_threads),
        "camera_frame": benchmark_latency(args.model, cam, args.latency_threads),
        "measured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "wall-clock on the development laptop; other jobs may share the CPU",
    }


def _figure_title(label: str, root: str, split: str, rep: dict[str, Any]) -> str:
    o = rep["overall"]
    return f"{label}\n{root} {split} (n={rep['n_images']}): mIoU {100 * (o['miou'] or 0):.1f}%, false-safe {100 * (o['false_safe_rate'] or 0):.1f}%"


def _write_figures(fig_prefix: Path, label: str, root: str, split: str, rep: dict[str, Any], qual: list) -> None:
    ttl = _figure_title(label, root, split, rep)
    plot_confusion(np.array(rep["overall"]["confusion"]), Path(f"{fig_prefix}_confusion_{split}.png"), ttl)
    if qual:
        plot_qualitative(qual, Path(f"{fig_prefix}_grid_{split}.png"), ttl)


def _figures_only(args: argparse.Namespace) -> int:
    """--figures-only: redraw figures from an existing JSON (grid images are re-predicted)."""
    d = json.loads(args.out.read_text(encoding="utf-8"))
    seg = Segmenter(args.model, intra_op_threads=args.threads)
    for sp, rep in d["splits"].items():
        samples = subset(list_samples(args.data_root, sp), rep.get("max_images") or 0, args.seed)
        if list_fingerprint(samples) != rep["file_list_sha"]:
            raise ValueError(f"{sp}: file list differs from the one evaluated in {args.out}")
        qual = []
        for i in qualitative_indices(len(samples), args.n_qualitative):
            rgb, gt = read_rgb(samples[i].image), read_mask(samples[i].mask)
            qual.append((rgb, gt, seg(rgb)[0]))
        _write_figures(args.fig_prefix, d.get("label") or args.model.stem, args.data_root.name, sp, rep, qual)
    LOG.info("figures redrawn from %s", args.out)
    return 0


def _update_latency(args: argparse.Namespace) -> int:
    """--latency-only: refresh latency_ms (and latency claims) of an existing results JSON."""
    if not args.out.exists():
        raise FileNotFoundError(f"--latency-only needs an existing {args.out}")
    d = json.loads(args.out.read_text(encoding="utf-8"))
    if Path(d["model"]["path"]).name != args.model.name:
        raise ValueError(f"{args.out} was produced by {d['model']['path']}, not {args.model}")
    d["latency_ms"] = _measure_latency(args)
    d["claims"] = _claims(d.get("label", ""), args.tag or args.out.stem, args.out, d["splits"], d["latency_ms"]["camera_frame"])
    args.out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    LOG.info("updated latency in %s", args.out)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate an exported ONNX terrain segmenter.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/rugd5"))
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--out", type=Path, required=True, help="results JSON path")
    ap.add_argument("--fig-prefix", type=Path, default=None, help="e.g. deck_assets/seg_zeroshot -> *_confusion_<split>.png, *_grid_<split>.png")
    ap.add_argument("--label", default="", help="human label stored in the JSON / figure titles")
    ap.add_argument("--tag", default=None, help="short id prefix for claims rows (default: out file stem)")
    ap.add_argument("--threads", type=int, default=2, help="ORT intra-op threads for the accuracy pass")
    ap.add_argument("--latency-threads", type=int, nargs="*", default=[2, 4])
    ap.add_argument("--max-images", type=int, default=0, help="seeded subset per split (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-qualitative", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="mark results as a SMOKE run (not a trained model)")
    ap.add_argument("--latency-only", action="store_true", help="re-measure latency only and update an existing --out JSON")
    ap.add_argument("--figures-only", action="store_true", help="redraw --fig-prefix figures from an existing --out JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import cv2
    import onnxruntime as ort

    cv2.setNumThreads(2)
    if args.latency_only:
        return _update_latency(args)
    if args.figures_only:
        if args.fig_prefix is None:
            raise ValueError("--figures-only needs --fig-prefix")
        return _figures_only(args)
    seg = Segmenter(args.model, intra_op_threads=args.threads)
    splits: dict[str, Any] = {}
    for sp in args.splits:
        rep, qual = evaluate_split(seg, args.data_root, sp, args.max_images, args.seed, args.n_qualitative)
        splits[sp] = rep
        o = rep["overall"]
        LOG.info("%s %s: mIoU=%.4f acc=%.4f false-safe=%s", args.model.name, sp, o["miou"] or float("nan"), o["pixel_acc"] or float("nan"), o["false_safe_rate"])
        if args.fig_prefix is not None:
            _write_figures(args.fig_prefix, args.label or args.model.stem, args.data_root.name, sp, rep, qual)
    latency = _measure_latency(args) if args.latency_threads else {}
    out = {
        "schema": SCHEMA,
        "label": args.label,
        "smoke": bool(args.smoke),
        "claim_label": "Tested",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "path": args.model.as_posix(),
            "sha256": sha256_file(args.model),
            "kind": seg.spec.kind,
            "input_hw": list(seg.spec.input_hw),
            "sidecar": seg.spec.meta,
        },
        "eval_threads": args.threads,
        "splits": splits,
        "latency_ms": latency,
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "onnxruntime": ort.__version__, "python": sys.version.split()[0]},
        "notes": "Evaluated at label resolution; per-class IoU None = class absent in GT and never predicted.",
    }
    out["claims"] = _claims(args.label, args.tag or args.out.stem, args.out, splits, latency.get("camera_frame", {}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
