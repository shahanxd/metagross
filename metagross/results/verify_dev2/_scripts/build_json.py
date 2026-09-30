"""Build results/verify_dev2/verify_dev2.json from the aggregate (verify_dev2_agg.json) + pytest log + golden-run notes.

Adds a ``claims`` list (ledger row shape: id, value, unit, label, source, note), all label Simulated except the
pytest count (Tested: it is a test-suite count, not a simulation result). Units as stated per claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Downloads\sih again\metagross")
VER = ROOT / "results" / "verify_dev2"
SRC = "results/verify_dev2/verify_dev2.json"


def main() -> None:
    agg = json.loads((VER / "verify_dev2_agg.json").read_text())
    extra = json.loads((VER / "verify_extra.json").read_text()) if (VER / "verify_extra.json").exists() else {}
    out = dict(agg)
    out.update(extra)
    claims: list[dict] = []

    def add(cid: str, value, unit: str, note: str, label: str = "Simulated", source: str = SRC) -> None:
        claims.append({"id": f"verify_dev2_{cid}", "value": value, "unit": unit, "label": label, "source": source, "note": note})

    how = ("verifier round-2 re-run: python -m metagross.sim.batch --split dev ({mode}, 5 Hz camera{cap}), "
           "config {cfg}, DEV seeds {seeds}, uncommitted tree on fee1da9, shared 4-core laptop (other agents + seg training)")
    for group, mode, cap in (("tier0", "--sensor-mode tier0, 2 workers, one stereo worker in parallel", ""),
                             ("stereo", "--sensor-mode stereo, FULL 1 worker in 3 batches 07:03-07:55 (load varied), T1 1-2 workers", ", --max-sim-s 150")):
        g = agg["groups"].get(group, {})
        for cfg in ("FULL", "T1"):
            a = g.get(cfg, {}).get("ALL")
            if not a:
                continue
            seeds = a["seeds"]
            note = how.format(mode=mode, cap=cap, cfg=cfg, seeds=",".join(map(str, seeds)))
            k = f"{group}_{cfg.lower()}"
            n = a["n"]
            add(f"{k}_success", f"{a['success']}/{n}", "runs", note + "; success = referee: GT body origin within 2 m of B")
            add(f"{k}_arrived_est", f"{a['arrived_est']}/{n}", "runs", note + "; ARRIVED commanded in the vehicle's own estimate (gt/cmds.npz)")
            add(f"{k}_arrived_short", f"{a['arrived_short']}/{n}", "runs", note + "; referee arrived_short: ARRIVED at rest with GT goal > 2 m away")
            add(f"{k}_ditch_entries", a["ditch"], "count", note)
            add(f"{k}_collisions", a["coll"], "count", note)
            add(f"{k}_water_entries", a["water"], "count", note)
            add(f"{k}_out_of_bounds", a["oob"], "runs", note)
            add(f"{k}_final_error_p50", round(a["final_error_p50"], 2), "m", note + "; GT distance to B at episode end")
            add(f"{k}_mean_speed", round(a["mean_speed"], 2), "m/s", note + "; mean over runs of referee mean_speed")
            add(f"{k}_compute_p50_p95", f"{a['compute_p50']:.0f} / {a['compute_p95']:.0f}", "ms", note + "; autonomy/timings.csv compute_ms over all ticks")
            if a.get("odo_ratio_p50") is not None:
                add(f"{k}_odometry_ratio_p50", round(a["odo_ratio_p50"], 3), "ratio",
                    note + "; estimated path length (telemetry pose, ~2 Hz) / GT path length at the same timestamps, median over runs")
            if a.get("pkt_med") is not None:
                add(f"{k}_packet_p50_p95", f"{a['pkt_med']:.0f} / {a['pkt_p95']:.0f}", "B",
                    note + f"; live telemetry packet_bytes; fraction > 600 B = {a['pkt_over_frac']:.3f}")
            h = g.get(cfg, {}).get("HOLDOUT")
            if h:
                add(f"{k}_holdout_success", f"{h['success']}/{h['n']}", "runs", note + f"; holdout subset seeds {h['seeds']}")
    for cid, c in extra.get("extra_claims", {}).items():
        add(cid, c["value"], c["unit"], c["note"], c.get("label", "Simulated"), c.get("source", SRC))
    out["claims"] = claims
    (VER / "verify_dev2.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{len(claims)} claims -> {VER / 'verify_dev2.json'}")


if __name__ == "__main__":
    sys.exit(main())
