"""Claims ledger: every number that appears on a slide, in the video or in the
README is registered here with its provenance and an honesty label.

Sources
-------
1. Any ``results/*.json`` **and** any ``results/<dir>/*.json`` (one level down, e.g.
   ``results/archive/verify_dev2/verify_dev2.json``) that carries a top-level ``"claims"`` list. Two row
   shapes are accepted: ``{id, value, label, source, note}`` and ``{claim, value, label, source}``.
   Top-level files are read first, then sub-directories, each in sorted order. An id is kept from
   the first file that defines it; a later file that repeats it is logged and skipped, so the
   ledger never holds duplicate ids and never silently overwrites a row.
2. The field extractors in :data:`EXTRACTORS`, for result files written before
   the ledger existed (they point at a JSON path and state the label). An extractor replaces an
   embedded row with the same id.
3. :func:`closed_loop_eval_rows`: stable ids for the final EVAL file
   ``results/closed_loop_eval.json`` (written by :mod:`metagross.sim.closed_loop_summary`), one row
   per ``closed_loop_eval_<mode>_<config>_<metric>`` (see :data:`EVAL_METRICS`), per family
   ``closed_loop_eval_<mode>_<config>_<family>_<metric>`` (:data:`FAMILY_METRIC_SUFFIXES`), one
   ``..._fail_<type>`` row per referee failure type, and the summed :data:`FAMILY_GROUPS`. The file's own
   ``claims`` list is ignored, because ``closed_loop_summary.make_claims`` names its rows
   ``closed_loop_<mode>_<config>_*``, the same ids as the DEV file. Nothing is emitted when the file
   is absent.

Output: ``results/claims.csv`` (id, value, unit, label, source, note) and a
``results/claims_index.json`` keyed by id, which the deck/video value fillers
read. Run ``python -m metagross.eval.claims`` after any evaluation, then
``python -m metagross.eval.results_md``.

Labels (CLAUDE.md): Tested (real data) · Simulated · Estimated · Proposed · Literature.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
LABELS = ("Tested", "Simulated", "Estimated", "Proposed", "Literature")

#: Final EVAL closed-loop file (written once, after the code freeze; see docs/EVAL_PREREGISTRATION.md).
CLOSED_LOOP_EVAL_FILE = "closed_loop_eval.json"
#: Files whose embedded ``claims`` list is not read (their rows come from a dedicated extractor).
EMBEDDED_CLAIMS_IGNORED = frozenset({CLOSED_LOOP_EVAL_FILE})
SENSOR_MODES = ("tier0", "stereo")


@dataclass(frozen=True)
class Extractor:
    id: str
    file: str  # relative to results/
    path: str  # dotted path inside the JSON
    label: str
    unit: str = ""
    fmt: str = "{:.3g}"
    note: str = ""


EXTRACTORS: tuple[Extractor, ...] = (
    # --- real-data stereo VO (KITTI odometry, camera only) ---------------------------------
    Extractor("kitti05_t_err_pct", "kitti_vo_05.json", "t_err_pct", "Tested", "%", "{:.2f}", "KITTI 05, camera-only stereo VO, KITTI protocol 100-800 m"),
    Extractor("kitti05_r_err_deg100m", "kitti_vo_05.json", "r_err_deg_per_100m", "Tested", "deg/100 m", "{:.2f}", "KITTI 05"),
    Extractor("kitti05_drift_1000m_pct", "kitti_vo_05.json", "drift_1000m_pct", "Tested", "%", "{:.2f}", "KITTI 05, mean over 1000 m segments"),
    Extractor("kitti05_drift_100m_pct", "kitti_vo_05.json", "drift_100m_pct", "Tested", "%", "{:.2f}", "KITTI 05"),
    Extractor("kitti05_drift_500m_pct", "kitti_vo_05.json", "drift_500m_pct", "Tested", "%", "{:.2f}", "KITTI 05"),
    Extractor("kitti07_t_err_pct", "kitti_vo_07.json", "t_err_pct", "Tested", "%", "{:.2f}", "KITTI 07, camera-only stereo VO"),
    Extractor("kitti07_r_err_deg100m", "kitti_vo_07.json", "r_err_deg_per_100m", "Tested", "deg/100 m", "{:.2f}", "KITTI 07"),
    Extractor("kitti07_drift_100m_pct", "kitti_vo_07.json", "drift_100m_pct", "Tested", "%", "{:.2f}", "KITTI 07"),
    Extractor("kitti07_drift_500m_pct", "kitti_vo_07.json", "drift_500m_pct", "Tested", "%", "{:.2f}", "KITTI 07"),
    # KITTI 00 ids come from kitti_summary.json's claims list, which carries the
    # "frames 1101-4540 only" caveat (the mirror's frames 0-1100 are another drive).
    # --- integrity monitor ------------------------------------------------------------------
    Extractor("integrity_auroc_07to05", "integrity_07to05.json", "meta.auroc_test", "Tested", "", "{:.3f}",
              "trained on degraded KITTI 07, tested on degraded KITTI 05 (real images, synthetic degradations)"),
    Extractor("integrity_auroc_silent_07to05", "integrity_07to05.json", "meta.auroc_test_silent_only", "Tested", "", "{:.3f}",
              "silent failures only; just 6 positives"),
    # --- runtime ----------------------------------------------------------------------------
    Extractor("localizer_hz_640", "localizer_timing.json", "hz_localizer_only", "Tested", "Hz", "{:.1f}", "Localizer.update at 640 px with shared disparity, i5-1135G7"),
    Extractor("renderer_sgbm_valid_ground", "renderer_bench.json", "sgbm.valid_ground_fraction", "Simulated", "", "{:.3f}", "rendered stereo, SGBM valid disparity on ground within 10 m"),
    Extractor("renderer_sgbm_median_err_px", "renderer_bench.json", "sgbm.median_abs_disp_err_px", "Simulated", "px", "{:.2f}", "rendered stereo, SGBM vs GT disparity on ground"),
    Extractor("renderer_ms_js_side", "renderer_bench.json", "stereo_ms_js_side.mean", "Simulated", "ms", "{:.1f}", "browser-side render+readback per stereo pair, Iris Xe"),
)


@dataclass(frozen=True)
class EvalMetric:
    """One ``closed_loop_eval_<mode>_<config>_<suffix>`` row: key in the ``all`` aggregate of
    :func:`metagross.sim.closed_loop_summary._agg`, unit and format."""
    suffix: str
    key: str
    unit: str
    fmt: str
    meaning: str


EVAL_METRICS: tuple[EvalMetric, ...] = (
    EvalMetric("n", "n", "runs", "{:d}", "number of EVAL runs"),
    EvalMetric("success", "n_success", "runs", "{:d}", "runs with the true goal within the mission success radius (referee)"),
    EvalMetric("success_rate", "success_rate", "fraction", "{:.3f}", "success / n"),
    EvalMetric("ditch", "ditch_entries", "count", "{:d}", "ditch entries (referee, GT)"),
    EvalMetric("collision", "collisions", "count", "{:d}", "collisions (referee, GT)"),
    EvalMetric("water", "water_entries", "count", "{:d}", "water entries (referee, GT)"),
    EvalMetric("oob", "out_of_bounds", "runs", "{:d}", "runs ended out of bounds"),
    EvalMetric("arrived_short", "arrived_short", "runs", "{:d}",
               "declared ARRIVED in its own pose estimate but ended outside the success radius"),
    EvalMetric("final_err_p50", "final_error_median_m", "m", "{:.2f}", "median GT distance to B at episode end"),
    EvalMetric("mean_speed", "mean_speed_mps", "m/s", "{:.2f}", "mean over runs of referee path length / episode time"),
    EvalMetric("false_stops", "false_stops", "count", "{:d}", "stops that ground truth does not justify (referee)"),
    EvalMetric("compute_p50", "compute_ms_p50", "ms", "{:.1f}",
               "median autonomy compute per 5 Hz tick (200 ms budget), 4 vCPU container, 4 episodes in parallel"),
    EvalMetric("compute_p95", "compute_ms_p95", "ms", "{:.1f}",
               "95th-percentile autonomy compute per 5 Hz tick (200 ms budget), 4 vCPU container, 4 episodes in parallel"),
)

#: Per-family rows ``closed_loop_eval_<mode>_<config>_<family>_<suffix>`` (subset of :data:`EVAL_METRICS`).
FAMILY_METRIC_SUFFIXES = frozenset({"n", "success", "ditch", "collision", "water", "oob", "arrived_short", "false_stops"})
#: Derived family groups quoted on the deck: summed ``n`` and ``n_success`` of the member families.
FAMILY_GROUPS: dict[str, tuple[str, ...]] = {"F2F3_ditch_crest": ("F2_ditch_field", "F3_crest_ditch")}

# seg_cpu: training finished if the last per-epoch validation line closes the planned iterations
SEG_CPU_FILE = "seg_cpu.json"
SEG_CPU_LOG = REPO / "runs" / "seg_cpu" / "train.log"
_EPOCH_VAL_RE = re.compile(r"train_seg: epoch (\d+): val mIoU")


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for key in dotted.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def claim_files(results_dir: Path = RESULTS) -> list[Path]:
    """``results/*.json`` (sorted), then ``results/<dir>/*.json`` (sorted), minus the ledger's own files."""
    top = sorted(results_dir.glob("*.json"))
    sub = sorted(p for p in results_dir.glob("*/*.json"))
    return [p for p in (*top, *sub) if not p.name.startswith("claims")]


def seg_cpu_training_complete(seg: dict, log_path: Path = SEG_CPU_LOG) -> Optional[str]:
    """Replacement note suffix if the CPU segmentation training ran to its planned end, else None.

    Complete means both: ``training.run_progress.complete`` is true in ``results/seg_cpu.json``,
    and ``runs/seg_cpu/train.log`` holds an ``epoch E: val`` line with (E+1) * epoch_iters >=
    total_iters_planned (the last epoch finished)."""
    tr = seg.get("training") or {}
    total, per_ep, best = tr.get("total_iters_planned"), tr.get("epoch_iters"), tr.get("iterations_at_best")
    if not (isinstance(total, int) and isinstance(per_ep, int) and isinstance(best, int)):
        return None
    if not (tr.get("run_progress") or {}).get("complete"):
        return None
    try:
        epochs = [int(m.group(1)) for m in _EPOCH_VAL_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace"))]
    except OSError:
        return None
    if not epochs or (max(epochs) + 1) * per_ep < total:
        return None
    return f"complete, best checkpoint at iteration {best}/{total}"


def _seg_cpu_note(note: str, done: Optional[str]) -> str:
    """'... best @ iter 2500/2750 PARTIAL' -> '... complete, best checkpoint at iteration 2500/2750'."""
    if not done or "PARTIAL" not in note:
        return note
    note = re.sub(r",?\s*best @ iter \d+/\d+\s*PARTIAL", f", {done}", note)
    return note.replace("PARTIAL", done)


def _embedded_rows(js: Path, results_dir: Path) -> list[dict[str, str]]:
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("skip %s: %s", js.name, exc)
        return []
    if not isinstance(data, dict):
        return []
    rel = js.relative_to(results_dir).as_posix()
    seg_done = seg_cpu_training_complete(data) if rel == SEG_CPU_FILE else None
    out = []
    for c in data.get("claims") or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id") or c.get("claim")
        if not cid:
            continue
        note = str(c.get("note", c.get("claim", "") if c.get("id") else ""))
        out.append({
            "id": str(cid),
            "value": str(c.get("value", "")),
            "unit": str(c.get("unit", "")),
            "label": str(c.get("label", "")),
            "source": str(c.get("source", f"results/{rel}")),
            "note": _seg_cpu_note(note, seg_done),
        })
    return out


def closed_loop_eval_rows(results_dir: Path = RESULTS) -> list[dict[str, str]]:
    """Rows ``closed_loop_eval_<mode>_<config>_<metric>`` from ``results/closed_loop_eval.json``
    (shape of :mod:`metagross.sim.closed_loop_summary`: ``doc[mode]['aggregate'][config]['all']``).
    Empty if the file is absent or unreadable; a missing metric is logged and skipped."""
    path = results_dir / CLOSED_LOOP_EVAL_FILE
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("skip %s: %s", path.name, exc)
        return []
    rows: list[dict[str, str]] = []
    for mode in SENSOR_MODES:
        agg = _dig(doc, f"{mode}.aggregate")
        if not isinstance(agg, dict):
            continue
        for cfg in sorted(agg):
            al = _dig(agg, f"{cfg}.all")
            if not isinstance(al, dict) or not al.get("n"):
                continue
            seeds = [int(s) for s in al.get("seeds") or []]
            seed_txt = f"seeds {min(seeds)}-{max(seeds)}" if seeds else "seeds not listed"
            base = (f"EVAL closed loop (pre-registered, docs/EVAL_PREREGISTRATION.md), sensor mode {mode}, config {cfg}, "
                    f"n={al['n']}, {seed_txt}, split={doc.get('split', '?')}; referee on GT")
            rows += _eval_metric_rows(al, EVAL_METRICS, f"closed_loop_eval_{mode}_{cfg}",
                                      f"{mode}.aggregate.{cfg}.all", base, f"{mode}/{cfg}")
            fams = _dig(agg, f"{cfg}.by_family") or {}
            fam_metrics = tuple(m for m in EVAL_METRICS if m.suffix in FAMILY_METRIC_SUFFIXES)
            for fam in sorted(fams):
                fb = fams[fam]
                if not isinstance(fb, dict) or not fb.get("n"):
                    continue
                rows += _eval_metric_rows(fb, fam_metrics, f"closed_loop_eval_{mode}_{cfg}_{fam}",
                                          f"{mode}.aggregate.{cfg}.by_family.{fam}", f"{base}; family {fam}",
                                          f"{mode}/{cfg}/{fam}")
            for group, members in FAMILY_GROUPS.items():
                parts = [fams.get(f) for f in members]
                if not all(isinstance(p, dict) and p.get("n") for p in parts):
                    continue
                n = sum(int(p["n"]) for p in parts)
                ok = sum(int(p["n_success"]) for p in parts)
                src = " + ".join(f"results/{CLOSED_LOOP_EVAL_FILE}#{mode}.aggregate.{cfg}.by_family.{f}" for f in members)
                for suffix, val, meaning in (("n", n, "number of EVAL runs"), ("success", ok, "runs that reached B")):
                    rows.append({"id": f"closed_loop_eval_{mode}_{cfg}_{group}_{suffix}", "value": str(val),
                                 "unit": "runs", "label": "Simulated", "source": src,
                                 "note": f"{base}; families {' + '.join(members)} summed; {meaning}"})
    return rows


def _eval_metric_rows(block: dict, metrics: tuple[EvalMetric, ...], id_prefix: str, json_path: str,
                      note: str, where: str) -> list[dict[str, str]]:
    """Rows for one aggregate block (``all`` or one family): the metrics, then one row per failure type."""
    rows: list[dict[str, str]] = []
    for m in metrics:
        val = block.get(m.key)
        if val is None:
            LOG.warning("closed_loop_eval %s: %s missing", where, m.key)
            continue
        try:
            text = m.fmt.format(int(val) if m.fmt == "{:d}" else float(val))
        except (TypeError, ValueError):
            text = str(val)
        rows.append({"id": f"{id_prefix}_{m.suffix}", "value": text, "unit": m.unit, "label": "Simulated",
                     "source": f"results/{CLOSED_LOOP_EVAL_FILE}#{json_path}.{m.key}", "note": f"{note}; {m.meaning}"})
    for ftype, count in sorted((block.get("failure_types") or {}).items()):
        if ftype in ("None", "null", ""):  # successes are the 'success' row
            continue
        rows.append({"id": f"{id_prefix}_fail_{ftype}", "value": str(int(count)), "unit": "runs", "label": "Simulated",
                     "source": f"results/{CLOSED_LOOP_EVAL_FILE}#{json_path}.failure_types.{ftype}",
                     "note": f"{note}; runs whose referee failure type is {ftype}"})
    return rows


def collect(results_dir: Path = RESULTS) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    origin: dict[str, str] = {}
    dups: list[str] = []

    def add(row: dict[str, str], where: str) -> None:
        if row["id"] in rows:
            dups.append(f"{row['id']} ({where}; kept {origin[row['id']]})")
            return
        rows[row["id"]] = row
        origin[row["id"]] = where

    for js in claim_files(results_dir):
        rel = js.relative_to(results_dir).as_posix()
        if rel in EMBEDDED_CLAIMS_IGNORED:
            continue
        for r in _embedded_rows(js, results_dir):
            add(r, rel)
    for ex in EXTRACTORS:
        path = results_dir / ex.file
        if not path.exists():
            continue
        val = _dig(json.loads(path.read_text(encoding="utf-8")), ex.path)
        if val is None:
            LOG.warning("extractor %s: %s#%s missing", ex.id, ex.file, ex.path)
            continue
        text = ex.fmt.format(val) if isinstance(val, (int, float)) else str(val)
        # an extractor deliberately replaces an embedded row with the same id (as before round 3):
        # it names the primary file and fixes the display format (e.g. KITTI 05/07 from kitti_summary.json)
        rows.pop(ex.id, None)
        origin.pop(ex.id, None)
        add({"id": ex.id, "value": text, "unit": ex.unit, "label": ex.label,
             "source": f"results/{ex.file}#{ex.path}", "note": ex.note}, f"extractor {ex.file}")
    for r in closed_loop_eval_rows(results_dir):
        add(r, CLOSED_LOOP_EVAL_FILE)
    if dups:
        LOG.warning("duplicate claim ids skipped (first definition kept): %s", dups)
    bad = [r["id"] for r in rows.values() if r["label"] not in LABELS]
    if bad:
        LOG.warning("claims with non-standard labels: %s", bad)
    return sorted(rows.values(), key=lambda r: r["id"])


def write(rows: list[dict[str, str]], results_dir: Path = RESULTS) -> Path:
    out = results_dir / "claims.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "value", "unit", "label", "source", "note"])
        w.writeheader()
        w.writerows(rows)
    (results_dir / "claims_index.json").write_text(json.dumps({r["id"]: r for r in rows}, indent=1), encoding="utf-8")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    rows = collect(args.results)
    out = write(rows, args.results)
    by_label: dict[str, int] = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
    LOG.info("wrote %s: %d claims %s", out, len(rows), by_label)


if __name__ == "__main__":
    main()
