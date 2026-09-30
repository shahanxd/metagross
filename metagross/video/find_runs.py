"""List run directories under a root, to pick the golden runs for the video.

    python -m video.find_runs results [--stereo-only] [--family F3]

One row per directory holding a ``result.json``: path, seed, family, config, sensor mode, success,
failure type, sim time [s], path length [m], ditch entries, debug bundles (and how many carry an
onboard image), telemetry packets. Rows are sorted so that stereo FULL / TYPICAL pairs of the same
seed sit next to each other. Offline tooling; reads only run outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def run_rows(root: str | Path, count_images: bool = False) -> list[dict[str, Any]]:
    """Summary rows for every ``result.json`` below ``root`` (``count_images`` opens every debug bundle)."""
    rows = []
    for res_path in sorted(Path(root).rglob("result.json")):
        run = res_path.parent
        try:
            res = json.loads(res_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        dbg = sorted((run / "autonomy" / "debug").glob("tick_*.npz"))
        n_left = sum(1 for f in dbg if _has_left(f)) if count_images else -1
        tel = run / "autonomy" / "telemetry.jsonl"
        rows.append({"path": run.as_posix(), "seed": res.get("seed"), "family": res.get("family", ""),
                     "config": (res.get("config") or {}).get("name", ""), "sensor": res.get("sensor_mode", ""),
                     "success": res.get("success"), "failure": res.get("failure_type") or "", "time_s": res.get("time"),
                     "path_m": res.get("path_length"), "ditch": res.get("ditch_entries"), "n_debug": len(dbg), "n_left": n_left,
                     "n_tel": sum(1 for _ in tel.open(encoding="utf-8")) if tel.exists() else 0})
    rows.sort(key=lambda r: (r["sensor"] != "stereo", str(r["seed"]), r["config"]))
    return rows


def _has_left(npz: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(npz) as z:
            return "extra_left_rgb.npy" in z.namelist()
    except zipfile.BadZipFile:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--stereo-only", action="store_true")
    ap.add_argument("--family", help="substring of the scenario family, e.g. F3")
    ap.add_argument("--images", action="store_true", help="count debug bundles with an onboard image (slower)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = run_rows(a.root, a.images)
    if a.stereo_only:
        rows = [r for r in rows if r["sensor"] == "stereo"]
    if a.family:
        rows = [r for r in rows if a.family in str(r["family"])]
    log.info(f"{'seed':>4} {'family':<18} {'config':<8} {'sensor':<6} {'ok':<5} {'failure':<10} {'time':>5} {'path':>6} "
             f"{'ditch':>5} {'dbg/img':>9} {'tel':>4}  path")
    for r in rows:
        log.info(f"{str(r['seed']):>4} {str(r['family']):<18} {r['config']:<8} {r['sensor']:<6} {str(r['success']):<5} "
                 f"{r['failure']:<10} {r['time_s'] or 0:5.1f} {r['path_m'] or 0:6.1f} {str(r['ditch']):>5} "
                 f"{r['n_debug']:>4}/{r['n_left']:<4} {r['n_tel']:>4}  {r['path']}")


if __name__ == "__main__":
    main()
