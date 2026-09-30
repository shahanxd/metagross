"""Fill a timeline's ``{placeholders}`` from the claims ledger and from run directories.

    python -m video.collect_values video/scenes/golden_demo.yaml --out video/out/values_golden.json \
        [--run run_ditch_full=results/golden/stereo/FULL/104]

The spec is the timeline YAML itself: its ``values:`` block maps each placeholder to exactly one
source, and ``{run_*}`` / ``{t0_*}`` references are filled from its ``runs:`` block (plus
``--run`` overrides) first::

    claims_index: results/claims_index.json      # written by python -m metagross.eval.claims
    values:
      kitti_t_err_pct: {claim: kitti_pooled_t_err_pct, fmt: "{:.1f}"}
      split_rcert_min_m: {run: runs/golden/f3_full, field: r_cert_m, stat: min, t0: 5, t1: 30, fmt: "{:.1f}"}
      glare_worst_mode: {run: runs/golden/f5_full, field: mode, stat: worst}
      packet_bytes: {run: runs/golden/golden_full, field: packet_bytes, stat: median, fmt: "{:.0f}"}
      final_error_m: {run: runs/golden/golden_full, json: result.json, key: final_error, fmt: "{:.1f}"}

Run fields (sampled over ``[t0, t1]`` run seconds, whole run if omitted):

* debug bundles (``autonomy/debug``): ``r_cert_m``, ``v_cap_mps``, ``q`` (health), ``speed_mps``
  (extras ``speed_meas``), ``mode`` (extras);
* telemetry (``autonomy/telemetry.jsonl``): ``packet_bytes``;
* ``json`` + ``key``: a dotted key of a JSON file in the run directory (``result.json``,
  ``mission.json``).

``stat`` is one of min / max / median / mean / first / last / worst (the most severe drive mode).

Output JSON: ``{"values": {name: {"value", "raw", "label", "source"}}, "claims": [...]}``. The
``claims`` list uses the ledger row shape (``id, value, label, source, note``) so the same file,
written under ``results/``, registers every run-derived number with ``metagross.eval.claims``.
Run-derived numbers are labelled ``Simulated``; ledger numbers keep their ledger label.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

log = logging.getLogger(__name__)

MODE_SEVERITY = ("ARRIVED", "HOLD", "NOMINAL", "CAUTION", "DEGRADED", "STOP_AND_LOOK", "SAFE_STOP")  # least -> most severe
DEBUG_FIELDS = ("r_cert_m", "v_cap_mps", "q", "speed_mps", "mode")
TELEMETRY_FIELDS = ("packet_bytes",)
RUN_LABEL = "Simulated"


def _dig(obj: Any, dotted: str) -> Any:
    for k in dotted.split("."):
        if not isinstance(obj, dict) or k not in obj:
            raise KeyError(dotted)
        obj = obj[k]
    return obj


def debug_series(run_dir: Path, field: str, t0: float = -np.inf, t1: float = np.inf) -> list[Any]:
    """Values of a debug-bundle field for bundles with t0 <= t <= t1 (time order)."""
    files = sorted((run_dir / "autonomy" / "debug").glob("tick_*.npz")) or sorted((run_dir / "autonomy").glob("debug_*.npz"))
    rows: list[tuple[float, Any]] = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            t = float(z["t"])
            if not (t0 <= t <= t1):
                continue
            extras = json.loads(str(z["extras_json"])) if "extras_json" in z.files else {}
            health = json.loads(str(z["health_json"])) if "health_json" in z.files else {}
            if field in ("r_cert_m", "v_cap_mps"):
                v: Any = float(z[field])
            elif field == "q":
                v = float(health.get("q", 1.0 - float(health.get("p_fail", 0.0))))
            elif field == "speed_mps":
                v = float(extras.get("speed_meas", extras.get("cmd_v", 0.0)))
            elif field == "mode":
                v = str(extras.get("mode", "HOLD"))
            else:
                raise KeyError(f"unknown debug field {field!r}")
            rows.append((t, v))
    rows.sort(key=lambda r: r[0])
    return [v for _, v in rows]


def telemetry_series(run_dir: Path, field: str, t0: float = -np.inf, t1: float = np.inf) -> list[Any]:
    p = run_dir / "autonomy" / "telemetry.jsonl"
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if t0 <= float(d["t"]) <= t1 and d.get(field) is not None:
                    out.append(d[field])
    return out


def reduce(values: list[Any], stat: str) -> Any:
    """Summary statistic of a series (see module docstring)."""
    if not values:
        raise ValueError("empty series")
    if stat == "worst":
        return max(values, key=lambda m: MODE_SEVERITY.index(m) if m in MODE_SEVERITY else -1)
    if stat == "first":
        return values[0]
    if stat == "last":
        return values[-1]
    a = np.asarray(values, np.float64)
    return float({"min": np.min, "max": np.max, "median": np.median, "mean": np.mean}[stat](a))


def spoken(v: Any) -> str:
    """Drive-mode names read aloud / on screen as plain words (STOP_AND_LOOK -> 'stop and look')."""
    return v.replace("_", " ").lower() if isinstance(v, str) and v in MODE_SEVERITY else str(v)


def collect_one(name: str, spec: dict, root: Path, claims: dict[str, dict]) -> dict:
    """One placeholder -> {'value', 'raw', 'label', 'source'}."""
    fmt: Optional[str] = spec.get("fmt")
    if "claim" in spec:
        row = claims.get(spec["claim"])
        if row is None:
            raise KeyError(f"{name}: claim {spec['claim']!r} not in the ledger")
        raw_s = str(row["value"]).split()[0]
        try:
            raw: Any = float(raw_s)
        except ValueError:
            raw = raw_s
        value = fmt.format(raw) if fmt and isinstance(raw, float) else raw_s
        return {"value": value, "raw": raw, "label": row.get("label", ""), "source": f"claims:{spec['claim']} ({row.get('source', '')})"}
    run = root / spec["run"]
    t0, t1 = float(spec.get("t0", -np.inf)), float(spec.get("t1", np.inf))
    if "json" in spec:
        raw = _dig(json.loads((run / spec["json"]).read_text(encoding="utf-8")), spec["key"])
        src = f"{spec['run']}/{spec['json']}#{spec['key']}"
    else:
        field, stat = spec["field"], spec.get("stat", "median")
        series = telemetry_series(run, field, t0, t1) if field in TELEMETRY_FIELDS else debug_series(run, field, t0, t1)
        raw = reduce(series, stat)
        src = f"{spec['run']} {field} {stat} over t=[{spec.get('t0', 'start')}, {spec.get('t1', 'end')}] s ({len(series)} samples)"
    value = fmt.format(raw) if fmt and isinstance(raw, (int, float)) else spoken(raw)
    return {"value": value, "raw": raw, "label": spec.get("label", RUN_LABEL), "source": src}


def collect(spec: dict, root: Path) -> dict:
    """Values spec -> {'values': {...}, 'claims': [...], 'missing': {...}} (missing = name -> reason)."""
    idx_path = root / spec.get("claims_index", "results/claims_index.json")
    claims = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    values: dict[str, dict] = {}
    missing: dict[str, str] = {}
    for name, s in (spec.get("values") or {}).items():
        try:
            values[name] = collect_one(name, s, root, claims)
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            missing[name] = f"{type(exc).__name__}: {exc}"
            log.warning("value %s not available: %s", name, missing[name])
    claim_rows = [{"id": f"video_{k}", "value": v["value"], "label": v["label"], "source": v["source"], "note": "demo video narration value"}
                  for k, v in values.items() if not v["source"].startswith("claims:")]
    return {"values": values, "claims": claim_rows, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="values spec YAML (e.g. video/scenes/values_golden.yaml)")
    ap.add_argument("--out", required=True, help="output JSON (video/out/... or results/video_values.json)")
    ap.add_argument("--root", default=".", help="base directory for run / claims paths")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH", help="override an entry of the 'runs:' block")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from video.render_timeline import fill, run_paths  # local: keeps this module importable without the renderer stack

    root = Path(a.root).resolve()
    spec = yaml.safe_load(Path(a.spec).read_text(encoding="utf-8"))
    spec = {**spec, "values": fill(spec.get("values") or {}, run_paths(spec, a.run))}
    res = collect(spec, root)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for k, v in res["values"].items():
        log.info("%-22s = %-10s [%s] %s", k, v["value"], v["label"], v["source"])
    if res["missing"]:
        log.warning("%d value(s) missing: %s", len(res["missing"]), ", ".join(res["missing"]))


if __name__ == "__main__":
    main()
