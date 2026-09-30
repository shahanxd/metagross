"""Offline link-budget report: replay real ``autonomy/telemetry.jsonl`` logs through the codec.

Reads only the autonomy's own decoded telemetry logs (no ground truth). For each log it
rebuilds every :class:`~metagross.contracts.messages.Telemetry` with
:func:`codec.telemetry_from_jsonable`, re-encodes it with the current codec and compares:

* ``before``: ``packet_bytes`` as logged by the run (codec v1: full 64 x 64 zlib costmap and
  free-text health keys);
* ``after``: the size of the current (v2, budget-laddered) packet.

It also streams both packet sequences through :class:`LinkEmulator` at the default radio
(``LINK_KBPS`` 9.6 kbit/s, ``LINK_LOSS`` 20 %, ``LINK_LATENCY_S`` 0.4 s) to show queueing, and
checks that decoding never *downgrades* a hazard cell (code >= ``U4_OCCLUDED``).

Note: the logged costmaps are themselves decoded v1 packets, which were lossless, so the replay
input equals what the vehicle produced.

CLI::

    python -m metagross.autonomy.link.budget_report --out results/link_budget.json \
        [--vendor results/link_replay/stereo_FULL_102.telemetry.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from metagross.autonomy.link import codec
from metagross.autonomy.link.emulator import LinkEmulator
from metagross.config.defaults import LINK_KBPS, LINK_LATENCY_S, LINK_LOSS, TELEMETRY_HZ

LOG = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
# Logs used while designing the ladder ("tuning") and logs from other code versions never looked at ("holdout").
TUNING_GLOBS: dict[str, str] = {
    "tier0_FULL": "results/runs_dev_tier0/FULL/*/autonomy/telemetry.jsonl",
    "tier0_TYPICAL": "results/runs_dev_tier0/TYPICAL/*/autonomy/telemetry.jsonl",
    "stereo_FULL": "results/archive/runs_dev_stereo/FULL/*/autonomy/telemetry.jsonl",
    "stereo_FULL_verify": "results/archive/verify_dev/stereo/FULL/*/autonomy/telemetry.jsonl",
}
HOLDOUT_GLOBS: dict[str, str] = {
    "holdout_verify_tier0": "results/archive/verify_dev/tier0/*/*/autonomy/telemetry.jsonl",
    "holdout_stereo_v2_nodyn": "results/archive/runs_dev_stereo/_v2_nodyn_FULL/*/autonomy/telemetry.jsonl",
    "holdout_tier0_v3_nodyn": "results/runs_dev_tier0/_v3_nodyn_FULL/*/autonomy/telemetry.jsonl",
    "holdout_integration": "results/archive/runs_integration/*/*/*/autonomy/telemetry.jsonl",
}
VENDOR_SOURCE_GLOB = "results/archive/verify_dev/stereo/FULL/102/autonomy/telemetry.jsonl"  # worst-case (stereo) real log
QUANTILES = (50, 95, 99, 100)
EMU_SEED = 0


def read_log(path: Path) -> list[dict]:
    """All JSON lines of one telemetry log."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def hazard_downgrades(original: np.ndarray, decoded: np.ndarray) -> int:
    """Number of cells whose original hazard code (>= U4_OCCLUDED) decoded to a less severe code."""
    haz = original >= codec.U4_OCCLUDED
    return int(np.count_nonzero(haz & (decoded < original)))


def _quant(v: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(v), float)
    if a.size == 0:
        return {}
    return {f"p{q}": round(float(np.percentile(a, q)), 1) for q in QUANTILES}


def _emulate(stream: list[tuple[float, int]]) -> dict[str, float]:
    """Stream (t, n_bytes) packets through the default link; latency quantiles and drop counts."""
    link = LinkEmulator(bandwidth_bps=LINK_KBPS * 1000.0, loss=LINK_LOSS, latency_s=LINK_LATENCY_S, seed=EMU_SEED)
    t_last = 0.0
    for t, n in stream:
        link.send(t, bytes(n))
        link.poll(t)
        t_last = t
    link.poll(t_last + 1e6)
    s = link.stats
    lat = np.asarray(s.latencies_s, float)
    return {"sent": s.sent, "delivered": s.delivered, "dropped_queue": s.dropped_queue, "dropped_loss": s.dropped_loss,
            "latency_p50_s": round(float(np.percentile(lat, 50)), 3) if lat.size else None,
            "latency_p95_s": round(float(np.percentile(lat, 95)), 3) if lat.size else None}


def analyse_group(files: list[Path]) -> dict:
    """Before/after sizes, ladder rungs, emulated link behaviour and hazard fidelity for a set of logs."""
    before, after, rungs = [], [], []
    downgrades = 0
    n_cm = 0
    emu_before, emu_after = [], []
    for f in files:
        sb, sa = [], []
        for d in read_log(f):
            tel = codec.telemetry_from_jsonable(d)
            pkt, info = codec.encode_with_info(tel)
            out = codec.decode(pkt)
            if tel.costmap_u4 is not None and out.costmap_u4 is not None:
                downgrades += hazard_downgrades(tel.costmap_u4, out.costmap_u4)
                n_cm += 1
            if "packet_bytes" in d:
                before.append(int(d["packet_bytes"]))
                sb.append((float(d["t"]), int(d["packet_bytes"])))
            after.append(len(pkt))
            sa.append((float(d["t"]), len(pkt)))
            rungs.append(info.rung)
        if sb:
            emu_before.append(_emulate(sb))
        emu_after.append(_emulate(sa))

    def _emu_sum(rows: list[dict]) -> dict:
        if not rows:
            return {}
        keys = ("sent", "delivered", "dropped_queue", "dropped_loss")
        out = {k: int(sum(r[k] for r in rows)) for k in keys}
        out["latency_p95_s_median_over_runs"] = round(float(np.median([r["latency_p95_s"] for r in rows if r["latency_p95_s"] is not None])), 3)
        out["latency_p95_s_max_over_runs"] = round(float(max(r["latency_p95_s"] for r in rows if r["latency_p95_s"] is not None)), 3)
        return out

    b, a = np.asarray(before, float), np.asarray(after, float)
    return {
        "n_logs": len(files), "n_packets": len(after), "n_costmaps": n_cm,
        "before_bytes": _quant(before), "after_bytes": _quant(after),
        "before_frac_over_budget": round(float(np.mean(b > codec.PACKET_BUDGET_B)), 4) if b.size else None,
        "after_frac_over_budget": round(float(np.mean(a > codec.PACKET_BUDGET_B)), 4) if a.size else None,
        "after_frac_over_target": round(float(np.mean(a > codec.PACKET_TARGET_B)), 4) if a.size else None,
        "before_mean_kbps": round(float(b.mean() * 8 * TELEMETRY_HZ / 1000.0), 2) if b.size else None,
        "after_mean_kbps": round(float(a.mean() * 8 * TELEMETRY_HZ / 1000.0), 2) if a.size else None,
        "rung_histogram": {str(k): int(v) for k, v in zip(*np.unique(rungs, return_counts=True))} if rungs else {},
        "hazard_cell_downgrades": downgrades,
        "emulated_link_before": _emu_sum(emu_before), "emulated_link_after": _emu_sum(emu_after),
    }


def _claims(groups: dict[str, dict]) -> list[dict]:
    src = "results/link_budget.json"
    note = f"codec v{codec.CODEC_VERSION}; budget {codec.PACKET_BUDGET_B} B = {LINK_KBPS} kbit/s at {TELEMETRY_HZ} Hz; replay of DEV telemetry.jsonl"
    out = []
    for g, r in groups.items():
        if not r.get("n_packets"):
            continue
        for when in ("before", "after"):
            q = r[f"{when}_bytes"]
            if not q:
                continue
            for p in ("p50", "p100"):
                out.append({"id": f"link_packet_{when}_{g}_{p}_bytes", "value": q[p], "unit": "B", "label": "Simulated",
                            "source": f"{src}#groups.{g}.{when}_bytes.{p}", "note": f"{note}; n={r['n_packets']} packets"})
        out.append({"id": f"link_packet_after_{g}_frac_over_budget", "value": r["after_frac_over_budget"], "unit": "fraction",
                    "label": "Simulated", "source": f"{src}#groups.{g}.after_frac_over_budget", "note": note})
    out.append({"id": "link_packet_budget_bytes", "value": codec.PACKET_BUDGET_B, "unit": "B", "label": "Proposed",
                "source": "metagross/autonomy/link/codec.py#PACKET_BUDGET_B", "note": "9.6 kbit/s / 8 / 2 Hz"})
    return out


def build_report(repo: Path = REPO, vendor_to: Optional[Path] = None) -> dict:
    """Run the replay over all known log locations under ``repo``."""
    groups: dict[str, dict] = {}
    for name, pattern in {**TUNING_GLOBS, **HOLDOUT_GLOBS}.items():
        files = sorted(repo.glob(pattern))
        if not files:
            LOG.warning("no logs for %s (%s)", name, pattern)
            continue
        LOG.info("%s: %d logs", name, len(files))
        groups[name] = analyse_group(files)
    if vendor_to is not None:
        src = sorted(repo.glob(VENDOR_SOURCE_GLOB))
        if src:
            vendor_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src[0], vendor_to)
            LOG.info("vendored %s -> %s", src[0], vendor_to)
    return {
        "codec_version": codec.CODEC_VERSION, "packet_budget_bytes": codec.PACKET_BUDGET_B, "packet_target_bytes": codec.PACKET_TARGET_B,
        "link": {"kbps": LINK_KBPS, "loss": LINK_LOSS, "latency_s": LINK_LATENCY_S, "telemetry_hz": TELEMETRY_HZ, "emulator_seed": EMU_SEED},
        "ladder": [{"rung": i, "pool": r.pool, "cell_m": codec.COSTMAP_RES_M * r.pool if r.pool else None,
                    "ground_levels": r.ground_levels if r.pool else None, "all_health": r.all_health} for i, r in enumerate(codec.LADDER)],
        "tuning_groups": list(TUNING_GLOBS), "holdout_groups": list(HOLDOUT_GLOBS),
        "groups": groups,
        "method": "before = packet_bytes logged by the DEV runs (codec v1); after = len(codec.encode(telemetry_from_jsonable(line))) "
                  "for every line; emulated link = LinkEmulator(9.6 kbit/s, 20 % loss, 0.4 s latency, seed 0) fed at the logged t.",
        "claims": _claims(groups),
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "link_budget.json")
    ap.add_argument("--vendor", type=Path, default=None, help="copy one real stereo log here as a test fixture")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rep = build_report(REPO, a.vendor)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    LOG.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
