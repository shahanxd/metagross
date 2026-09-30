"""Summarise batch runs: result + last telemetry reasons + events (sim-side analysis)."""
import json, sys, collections
from pathlib import Path

root = Path(sys.argv[1])
seeds = sys.argv[2:] if len(sys.argv) > 2 else None
for d in sorted(root.iterdir()):
    if not d.is_dir() or (seeds and d.name.lstrip("0") not in seeds and d.name not in seeds):
        continue
    rp = d / "result.json"
    if not rp.exists():
        continue
    r = json.loads(rp.read_text())
    tel = [json.loads(l) for l in open(d / "autonomy" / "telemetry.jsonl")] if (d / "autonomy" / "telemetry.jsonl").exists() else []
    reasons = collections.Counter((t.get("mode"), (t.get("reason") or "")[:18]) for t in tel)
    arrived = any(t.get("mode") == "ARRIVED" for t in tel)
    ev = json.loads((d / "gt" / "events.json").read_text())
    evs = [(e["t"], e["type"], {k: v for k, v in e.items() if k not in ("t", "type")}) for e in ev if e["type"] not in ("lighting_on", "lighting_off")]
    last = tel[-1] if tel else {}
    print(f"{d.name} {r['family'][:12]:12s} ok={r['success']!s:5s} fail={r['failure_type']} t={r['time']} L={r['path_length']} v={r['mean_speed']} "
          f"ferr={r['final_error']} ditch={r['ditch_entries']} coll={r['collisions']} fs={r['false_stops']} arrivedEst={arrived} "
          f"cms={r['compute_ms_mean']}")
    print("    last:", last.get("mode"), last.get("reason"), "pose", [round(v, 1) for v in last.get("pose_xy_yaw", [])], "| events", evs[:4])
    print("    reasons:", reasons.most_common(6))
