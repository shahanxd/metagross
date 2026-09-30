"""Operator-console screenshots from REPLAYS of real logged EVAL runs (tier-0 simulation).

Serves the repo root on 127.0.0.1 (free port), opens operator_ui/index.html in Playwright's bundled
headless Chromium at a 1920x1080 viewport (device scale 1.5 -> 2880x1620 px), loads a FULL run's
mission.json + autonomy/telemetry.jsonl through the console's own ?mission=&replay= URL parameters
(deterministic=1: no wall clock, packets applied immediately), then calls the console's
window.consoleSeek(t) to show the state at chosen mission times. Nothing in the console is altered;
a thin provenance strip is appended *below* the screenshot with Pillow.

The console is NOT live here: every frame is a replay of a telemetry log written during a closed-loop
simulation run (tier-0 synthetic depth sensor, no camera images, so the VO tile reads LOST and
localisation is wheel + gyro, per docs/SIMULATION.md).

Run from the repo root:  python deck_assets/final/src/console_shots.py
Outputs: deck_assets/final/console_<k>.png and deck_assets/final/console.json
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "deck_assets" / "final"
RUNS = "results/runs_eval_tier0/FULL"
VIEWPORT = {"width": 1920, "height": 1080}
DSF = 1.5  # device scale factor -> 2880 x 1620 px screenshots
STRIP_H = 64  # px, provenance strip appended below the screenshot (at saved resolution)
GREY = (110, 110, 110)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# (k, EVAL seed, mission time t [s] to seek to, map extent [m], why this moment)
MOMENTS = [
    (1, 44, 22.8, 16, "F3 crest + ditch: crest-shadow (unseen) cells ahead, ditch cells passed; the governor is limited by "
                      "the visible range (GOV_RVIS R=3.0 m -> v_cap 1.98 m/s)"),
    (2, 48, 9.6, 16, "F1 trail: STOP_AND_LOOK 2/3 with lethal cells directly ahead, v_cap 0 (referee: stop justified, "
                     "reason hazard_ahead)"),
    (3, 48, 26.4, 24, "F1 trail: after the look it detoured around the hazard and is back to NOMINAL; the event log "
                      "shows NOMINAL -> STOP_AND_LOOK -> NOMINAL (RESUME after look 1)"),
]

DOM_FIELDS = {
    "mission": "mission-id", "mode": "mode-chip", "reason": "mode-reason", "in_mode": "mode-since",
    "dist_to_b": "dist-goal", "clock": "clock-mission", "speed_vcap": "speed-val", "r_cert": "rcert-val",
    "q": "q-text", "p_fail": "pfail-val", "pos_sigma": "sigma-val", "vo": "vo-val", "slip": "slip-val",
    "link_kbps": "link-kbps", "link_loss": "link-loss", "packets": "pk-n", "pk_last": "pk-last",
    "pk_mean": "pk-mean", "pk_lost": "pk-lost",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def packet_at(seed: int, t: float) -> dict:
    """Last telemetry packet with t_pkt <= t (what consoleSeek(t) ends on)."""
    rows = [json.loads(l) for l in (REPO / RUNS / f"{seed:03d}" / "autonomy" / "telemetry.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["t"] <= t + 1e-9]
    r = rows[-1]
    return {k: r[k] for k in ("t", "seq", "mode", "reason", "pose", "pos_sigma_m", "v_cap_mps", "r_cert_m", "speed_mps")} | {
        "health.q": r["health"].get("q"), "health.r_vis_m": r["health"].get("r_vis_m"), "health.vo_ok": r["health"].get("vo_ok"),
        "packet_bytes": r.get("packet_bytes"), "n_packets_applied": len(rows)}


def add_strip(png: Path, text: str) -> tuple[int, int]:
    im = Image.open(png).convert("RGB")
    out = Image.new("RGB", (im.width, im.height + STRIP_H), (255, 255, 255))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((24, im.height + STRIP_H // 2), text, fill=GREY, font=ImageFont.truetype(FONT, 30), anchor="lm")
    out.save(png, optimize=True)
    return out.size


def main() -> int:
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(REPO)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    records, logs = [], []
    try:
        time.sleep(1.0)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            for k, seed, t, extent, why in MOMENTS:
                run = f"/{RUNS}/{seed:03d}"
                res = json.loads((REPO / RUNS / f"{seed:03d}" / "result.json").read_text())
                page = browser.new_page(viewport=VIEWPORT, device_scale_factor=DSF)
                page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
                page.goto(f"http://127.0.0.1:{port}/operator_ui/index.html?deterministic=1"
                          f"&mission={run}/mission.json&replay={run}/autonomy/telemetry.jsonl")
                page.wait_for_function("window.__consoleReady === true", timeout=15000)
                page.wait_for_function("document.getElementById('pk-n').textContent !== '0'", timeout=15000)
                page.click(f"#zoom button[data-extent='{extent}']")
                page.evaluate(f"window.consoleSeek({t})")
                page.evaluate("window.dispatchEvent(new Event('resize'))")
                time.sleep(0.5)
                dom = page.evaluate("(ids) => Object.fromEntries(Object.entries(ids).map(([k, id]) => [k, document.getElementById(id).textContent]))", DOM_FIELDS)
                dom["replay_badge_visible"] = page.evaluate("!document.getElementById('replay-badge').hidden")
                dom["event_log"] = page.evaluate("Array.from(document.querySelectorAll('#log li')).map(li => li.innerText.replace(/\\n/g, ' | '))")
                png = OUT / f"console_{k}.png"
                page.screenshot(path=str(png))
                page.close()
                prov = (f"Simulated run, replayed (not live) · tier-0, EVAL seed {seed} ({res['family']}), FULL, "
                        f"t = {t:.1f} s · {RUNS}/{seed:03d}/autonomy/telemetry.jsonl")
                size = add_strip(png, prov)
                records.append({
                    "file": str(png.relative_to(REPO)) if png.is_relative_to(REPO) else str(png), "size_px": list(size), "label": "Simulated run, replayed",
                    "moment": why, "seed": seed, "family": res["family"], "split": res["split"], "sensor_mode": res["sensor_mode"],
                    "seek_t_s": t, "map_extent_m": extent, "provenance_line": prov,
                    "sources": {"mission": f"{RUNS}/{seed:03d}/mission.json", "telemetry": f"{RUNS}/{seed:03d}/autonomy/telemetry.jsonl",
                                "result": f"{RUNS}/{seed:03d}/result.json"},
                    "run_outcome": {kk: res[kk] for kk in ("success", "failure_type", "time", "path_length", "spl", "final_error", "stops", "min_clearance")},
                    "telemetry_packet_shown": packet_at(seed, t),
                    "console_readout": dom,
                    "how": "Every value on screen is rendered by operator_ui/app.js from the telemetry packets with t <= seek_t "
                           "(window.consoleSeek); link kbps/loss are computed by app.js over the replayed packets' packet_bytes/seq.",
                })
            browser.close()
    finally:
        srv.terminate()
        srv.wait(timeout=5)
    (OUT / "console.json").write_text(json.dumps({
        "generator": "deck_assets/final/src/console_shots.py",
        "what": "Operator console (operator_ui/) replaying real logged EVAL closed-loop runs; replay only, not a live link.",
        "caveats": ["tier-0 synthetic depth sensor: no camera images, so VO cannot run (VO tile reads LOST; localisation is wheel + gyro, docs/SIMULATION.md)",
                    "the console is replay-only today (live link not built; docs/PIPELINE_STATUS.md)"],
        "viewport": VIEWPORT, "device_scale_factor": DSF, "figures": records, "browser_errors": logs}, indent=1))
    for r in records:
        print(r["file"], r["size_px"], r["console_readout"]["mode"], r["console_readout"]["reason"])
    print("\n".join(logs) or "no page errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
