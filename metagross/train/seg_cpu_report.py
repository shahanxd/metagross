"""Post-process the CPU-trained segmenter's evaluation: provenance, baseline comparison, claims.

Input is the ``metagross.seg_eval/1`` JSON written by :mod:`metagross.eval.seg_eval` for
``models/lraspp_offroad5_cpu.onnx`` (RUGD-5L val + test at label resolution). This module
adds, in place:

* ``training`` - which checkpoint was evaluated (iteration at best / planned, partial or
  complete, sampler, repeat factors, the val subset used for checkpoint selection);
* ``comparison`` - per split, the same metrics for each baseline results file
  (``results/seg_smoke.json``, ``results/seg_zeroshot.json``) and ``ours - baseline``
  deltas, only where both were evaluated on the identical file list (SHA match);
* extra ``claims`` rows (label ``Tested``: RUGD is real data) - per-class IoU,
  false-safe split by hazard class, false-hazard rate, deltas vs. baselines, training
  iterations. Rows already written by ``seg_eval`` are kept.

All rates are fractions in [0, 1]; deltas are absolute differences of fractions.

    python -m metagross.train.seg_cpu_report --result results/seg_cpu.json \
        --ckpt runs/seg_cpu/finish/best_snapshot.pt --run-dir runs/seg_cpu \
        --baselines results/seg_smoke.json results/seg_zeroshot.json
    python -m metagross.train.seg_cpu_report --label-only --ckpt ... --run-dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from metagross.autonomy.perception.semantics import HAZARD_CLASSES, SEM_OBSTACLE, SEM_STABLE, SEM_WATER

LOG = logging.getLogger("seg_cpu_report")
TAG = "seg_cpu"
CLAIM_LABEL = "Tested"
SCALAR_KEYS = ("miou", "pixel_acc", "false_safe_rate", "false_safe_obstacle", "false_safe_water", "false_hazard_rate",
               "hazard_to_stable_rate", "obstacle_to_stable_rate", "water_to_stable_rate")
EXTRA_CLAIM_KEYS = ("false_safe_obstacle", "false_safe_water", "false_hazard_rate",
                    "hazard_to_stable_rate", "obstacle_to_stable_rate", "water_to_stable_rate")
DELTA_KEYS = ("miou", "false_safe_rate", "false_safe_water", "hazard_to_stable_rate")
CLASS_SHORT = {  # class name in seg_eval JSON -> short claim-id token
    "background/sky": "sky",
    "obstacle": "obstacle",
    "water/mud": "water",
    "unstable (grass/dirt)": "unstable",
    "stable (asphalt/concrete/gravel path)": "stable",
}
ROUND = 4  # decimals of fractions in claims
LATENCY_CAVEAT = ("shared laptop CPU (other jobs may be running): compare latencies only within one measurement "
                  "session, e.g. seg_eval --latency-only on both models back to back")


def checkpoint_info(ckpt: dict[str, Any], run_metrics: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Provenance of an evaluated checkpoint (``best.pt`` dict) + the run's ``metrics.json``."""
    meta = ckpt.get("meta", {})
    args = meta.get("args", {})
    planned = int(meta.get("total_iters") or 0)
    at_best = int(ckpt.get("global_iter") or 0)
    info: dict[str, Any] = {
        "iterations_at_best": at_best,
        "total_iters_planned": planned,
        "partial": bool(planned and at_best < planned),
        "best_epoch": ckpt.get("epoch"),
        "epoch_iters": meta.get("iters_per_epoch"),
        "batch_size": args.get("bs"),
        "img_hw": meta.get("img_hw"),
        "aug": args.get("aug"),
        "sampling": args.get("sampling", "uniform"),
        "repeat_factor_sampling": meta.get("repeat_factor_sampling"),
        "lr": args.get("lr"),
        "n_train_images": meta.get("n_train"),
        "selection": f"best of per-epoch val mIoU on a seeded {meta.get('n_val')}-image subset of RUGD-5L val "
        "(so val numbers carry a small selection bias; test is untouched)",
        "val_at_selection": {k: (ckpt.get("val") or {}).get(k) for k in ("miou", "false_safe_rate")},
    }
    if run_metrics:
        info["run_progress"] = {
            "global_iter": run_metrics.get("global_iter"),
            "complete": run_metrics.get("complete"),
            "stopped_by_deadline": run_metrics.get("stopped_by_deadline"),
            "n_epochs_done": len(run_metrics.get("history", [])),
            "last_train_s_per_iter": (run_metrics.get("history") or [{}])[-1].get("train_s_per_iter"),
        }
    return info


def describe(info: dict[str, Any]) -> str:
    """One-line human label for figures / results."""
    part = " PARTIAL" if info["partial"] else ""
    return (f"LR-ASPP CPU-trained on RUGD-5L ({info['aug']} aug, {info['sampling']} sampling), "
            f"best @ iter {info['iterations_at_best']}/{info['total_iters_planned']}{part}")


def _rate(num: float, den: float) -> Optional[float]:
    return float(num / den) if den > 0 else None


def hazard_to_stable(confusion: Sequence[Sequence[int]]) -> dict[str, Optional[float]]:
    """Strict false-safe: share of GT hazard pixels predicted *stable path* (class 4) only.

    ``seg_eval``'s ``false_safe_rate`` counts hazard -> any traversable class (3 or 4);
    this is the subset the planner treats as free, low-cost ground.
    """
    cm = np.asarray(confusion, dtype=np.int64)
    hz = list(HAZARD_CLASSES)
    return {
        "hazard_to_stable_rate": _rate(cm[hz, SEM_STABLE].sum(), cm[hz].sum()),
        "obstacle_to_stable_rate": _rate(cm[SEM_OBSTACLE, SEM_STABLE], cm[SEM_OBSTACLE].sum()),
        "water_to_stable_rate": _rate(cm[SEM_WATER, SEM_STABLE], cm[SEM_WATER].sum()),
    }


def _overall_plus(overall: dict[str, Any]) -> dict[str, Any]:
    """seg_eval overall metrics + the strict hazard -> stable rates (from its confusion matrix)."""
    return dict(overall, **hazard_to_stable(overall["confusion"])) if "confusion" in overall else dict(overall)


def _scalars(overall: dict[str, Any]) -> dict[str, Any]:
    o = _overall_plus(overall)
    out = {k: o.get(k) for k in SCALAR_KEYS}
    out["per_class_iou"] = o.get("per_class_iou")
    return out


def compare(ours: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per split: our scalars, each baseline's scalars and ``ours - baseline`` on identical images."""
    comp: dict[str, Any] = {}
    for sp, rep in ours["splits"].items():
        row: dict[str, Any] = {"ours": dict(_scalars(rep["overall"]), n_images=rep["n_images"]), "baselines": {}}
        for name, b in baselines.items():
            brep = b.get("splits", {}).get(sp)
            if brep is None:
                continue
            same = brep.get("file_list_sha") == rep.get("file_list_sha")
            entry: dict[str, Any] = dict(_scalars(brep["overall"]), n_images=brep["n_images"], same_images=same,
                                         model=b.get("model", {}).get("path"), label=b.get("label"))
            if same:
                mine, theirs = row["ours"], entry
                entry["delta_ours_minus_baseline"] = {
                    k: (None if mine.get(k) is None or theirs.get(k) is None else mine[k] - theirs[k]) for k in SCALAR_KEYS
                }
            row["baselines"][name] = entry
        comp[sp] = row
    return comp


def extra_claims(result: dict[str, Any], comp: dict[str, Any], info: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Additional Tested claims (per-class IoU, false-safe by hazard, deltas, iterations)."""
    note_base = describe(info)
    rows: list[dict[str, Any]] = []

    def add(cid: str, value: Any, unit: str, note: str) -> None:
        if value is None:
            return
        rows.append({"id": cid, "value": round(value, ROUND) if isinstance(value, float) else value, "unit": unit,
                     "label": CLAIM_LABEL, "source": source, "note": note})

    for sp, rep in result["splits"].items():
        o = _overall_plus(rep["overall"])
        sub = f", seeded subset of {rep['max_images']} images" if rep.get("max_images") else ""
        note = f"{note_base}; RUGD-5L {sp} (n={rep['n_images']}{sub}), label resolution"
        for k in EXTRA_CLAIM_KEYS:
            add(f"{TAG}_{sp}_{k}", o.get(k), "fraction", note)
        for cname, short in CLASS_SHORT.items():
            add(f"{TAG}_{sp}_iou_{short}", (o.get("per_class_iou") or {}).get(cname), "fraction", note)
        for bname, b in comp[sp]["baselines"].items():
            d = b.get("delta_ours_minus_baseline")
            if not d:
                continue
            for k in DELTA_KEYS:
                add(f"{TAG}_{sp}_{k}_delta_vs_{bname}", d.get(k), "fraction",
                    f"{note}; ours minus {b.get('label') or bname} on the same {rep['n_images']} images")
    add(f"{TAG}_train_iters_at_best", info["iterations_at_best"], "iterations",
        f"{note_base}; batch {info['batch_size']}, {info['img_hw']} crops, laptop CPU (2 threads)")
    return rows


def update_result(result_path: Path, ckpt_path: Path, run_dir: Path, baseline_paths: Sequence[Path]) -> dict[str, Any]:
    """Add training / comparison / claims to ``result_path`` in place; returns the new dict."""
    import torch

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "metagross.seg_eval/1":
        raise ValueError(f"{result_path} is not a seg_eval result")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck.pop("model", None)
    mpath = run_dir / "metrics.json"
    run_metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    info = checkpoint_info(ck, run_metrics)
    baselines = {}
    for p in baseline_paths:
        if p.exists():
            baselines[p.stem.replace("seg_", "")] = json.loads(p.read_text(encoding="utf-8"))
        else:
            LOG.warning("baseline %s missing, skipped", p)
    comp = compare(result, baselines)
    source = result_path.as_posix()
    new = extra_claims(result, comp, info, source)
    new_ids = {c["id"] for c in new}  # re-running replaces our own rows instead of keeping stale ones
    base_claims = [c for c in result.get("claims", []) if c.get("id") not in new_ids]
    note = describe(info)
    for c in base_claims:  # seg_eval rows: make the checkpoint status explicit in the note
        if note not in str(c.get("note", "")):
            c["note"] = f"{note}; {c.get('note', '')}".rstrip("; ")
        if "_latency_" in str(c.get("id", "")) and LATENCY_CAVEAT not in c["note"]:
            c["note"] = f"{c['note']}; {LATENCY_CAVEAT}"
    result.update(training=info, comparison=comp, claims=base_claims + new)
    result_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for sp, row in comp.items():
        o = row["ours"]
        LOG.info("%s: mIoU %.4f false-safe %.4f (obstacle %s, water %s)", sp, o["miou"] or 0, o["false_safe_rate"] or 0, o["false_safe_obstacle"], o["false_safe_water"])
        for bname, b in row["baselines"].items():
            LOG.info("   vs %s: mIoU %.4f false-safe %.4f same_images=%s", bname, b["miou"] or 0, b["false_safe_rate"] or 0, b["same_images"])
    LOG.info("wrote %s (%d claims)", result_path, len(result["claims"]))
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", type=Path, help="seg_eval JSON to update in place")
    ap.add_argument("--ckpt", type=Path, required=True, help="the evaluated checkpoint (best.pt snapshot)")
    ap.add_argument("--run-dir", type=Path, required=True, help="training output dir (metrics.json)")
    ap.add_argument("--baselines", type=Path, nargs="*", default=[])
    ap.add_argument("--label-only", action="store_true", help="print the one-line label for --ckpt and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.label_only:
        import torch

        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck.pop("model", None)
        sys.stdout.write(describe(checkpoint_info(ck, None)) + "\n")
        return 0
    if args.result is None:
        ap.error("--result is required unless --label-only")
    update_result(args.result, args.ckpt, args.run_dir, args.baselines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
