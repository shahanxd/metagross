"""GO/NO-GO benchmark for the Three.js stereo renderer.

Renders N stereo pairs along a route, measures wall-clock time per pair INCLUDING
the Playwright transfer and Python decode, and checks stereo realism by running
OpenCV SGBM on the rendered pairs against the renderer's GT depth.

GO criteria (task B-renderer):
  * mean <= 120 ms per stereo pair (incl. decode),
  * WebGL renderer is hardware (not SwiftShader),
  * >= 60 % valid SGBM disparity on ground pixels within 10 m.

Outputs: results/renderer_bench.json, deck_assets/render_check.png.
Run (PowerShell; launches Chrome):
  & .venv\\Scripts\\python.exe scripts\\bench_renderer.py --n 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from metagross.config import defaults  # noqa: E402
from metagross.contracts.interfaces import SEM_CLASSES  # noqa: E402
from metagross.sim.render.bridge import ThreeRenderer  # noqa: E402
from metagross.sim.render.testscene import make_test_scenario, route_poses, terrain_height  # noqa: E402

log = logging.getLogger("bench_renderer")

GO_MS_PER_PAIR = 120.0
GO_VALID_FRAC = 0.60
GROUND_RANGE_M = 10.0
GROUND_SEM = (2, 3, 4)  # water/mud, unstable, stable (terrain classes of the 5-class scheme)
SGBM_NUM_DISP = 64
SGBM_BLOCK = 5
SEM_PALETTE = np.array([[135, 190, 235], [220, 50, 47], [20, 140, 190], [110, 170, 60], [200, 190, 150]], np.uint8)


def make_sgbm() -> cv2.StereoSGBM:
    """SGBM as specified for the bench: 3-way mode, 64 disparities, 5x5 blocks (grayscale input)."""
    b = SGBM_BLOCK
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=SGBM_NUM_DISP, blockSize=b, P1=8 * b * b, P2=32 * b * b,
                                 disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def build_scenario(seed: int) -> tuple[dict, str]:
    """Official scenario generator if another module already provides it, else the renderer's synthetic scene."""
    try:
        from metagross.sim.scenario import make_scenario  # type: ignore[attr-defined]

        return make_scenario(seed), "metagross.sim.scenario.make_scenario"
    except Exception as e:  # noqa: BLE001 - module may not exist yet
        log.info("official scenario generator unavailable (%s); using synthetic test scene", e)
        return make_test_scenario(seed), "metagross.sim.render.testscene.make_test_scenario"


def poses_for(scenario: dict, n: int, synthetic: bool) -> list[list[float]]:
    if synthetic:
        return route_poses(scenario, n, 0.0, 22.0)
    (x0, y0), (x1, y1) = scenario["start"]["xy"], scenario["goal"]["xy"]
    yaw = math.atan2(y1 - y0, x1 - x0)
    out = []
    for k in range(n):
        s = 0.6 * k / max(n - 1, 1)
        x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
        out.append([x, y, terrain_height(scenario, x, y), 0.0, 0.0, yaw])
    return out


def dynamic_state(scenario: dict, t: float) -> list[dict]:
    """Move each dynamic actor along its path from t = 0 (bench only; the world owns real triggering)."""
    out = []
    for d in scenario.get("dynamic", []):
        (ax, ay), (bx, by) = d["path"][0], d["path"][1]
        L = math.hypot(bx - ax, by - ay) or 1.0
        s = min(1.0, t * d.get("speed", 1.0) / L)
        out.append({"xy": [ax + s * (bx - ax), ay + s * (by - ay)], "yaw": math.atan2(by - ay, bx - ax)})
    return out


def colorize(a: np.ndarray, vmax: float, invalid: np.ndarray) -> np.ndarray:
    v = np.clip(a / vmax * 255.0, 0, 255).astype(np.uint8)
    c = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    c[invalid] = 0
    return c


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


EVENT_OFFSETS_S = (-0.6, 0.4, 1.4, 3.0)  # strip frames relative to event onset t0 (s)


def machine_cpu_load_pct() -> float | None:
    """Whole-machine CPU load (%) on Windows via CIM; None elsewhere or on failure. Recorded
    because the Playwright transfer (Node driver + CDP pipe) is CPU-bound and this laptop is shared."""
    import subprocess

    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).LoadPercentage"],
                             capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def render_events(r: ThreeRenderer, scenario: dict, pose: list[float], sgbm: cv2.StereoSGBM, fB: float) -> tuple[list[dict], list[np.ndarray]]:
    """For each scenario lighting event, render a 5 Hz sequence from t0-1 s to t0+3 s at a fixed
    pose (AE lag and noise evolve with dt), logging exposure, mean luma and SGBM ground validity."""
    out, rows = [], []
    dt = 1.0 / defaults.CAMERA_HZ_BATCH
    for ev in scenario.get("lighting", {}).get("events", []):
        t0 = float(ev["t0"])
        seq, strip = [], []
        n = int(round(4.0 / dt)) + 1
        for i in range(n):
            t = t0 - 1.0 + i * dt
            state = {"pose": pose, "t": t, "seq": 10_000 + i}
            left, right = r.render_stereo(state)
            gt = r.render_gt(state)
            disp = sgbm.compute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), right).astype(np.float32) / 16.0
            ground = np.isin(gt["semantic"], GROUND_SEM) & (gt["depth"] <= GROUND_RANGE_M)
            ground[:, :SGBM_NUM_DISP] = False
            vf = float((ground & (disp > 0)).sum()) / max(int(ground.sum()), 1)
            ex = r.last_js.get("exposure", {})
            seq.append({"t": round(t, 3), "mean_luma": float(right.mean()), "valid_ground_frac": vf,
                        **{k: ex.get(k) for k in ("E", "irr", "glare", "dust", "noise_dn")}})
            for off in EVENT_OFFSETS_S:
                if abs(t - (t0 + off)) < dt / 2:
                    strip.append(label(cv2.resize(left, (cal_w := left.shape[1] // 2, left.shape[0] // 2), interpolation=cv2.INTER_AREA),
                                       f"{ev['type']} t0{off:+.1f}s E={ex.get('E', 0):.2f} valid={vf:.0%}"))
        out.append({"event": ev, "sequence": seq})
        if strip:
            rows.append(np.concatenate(strip, axis=1))
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="stereo pairs to render")
    ap.add_argument("--seed", type=int, default=100, help="scenario seed (DEV seeds only: 100-129)")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--rgb", action="store_true", help="exact RGB transport instead of YUV 4:2:0")
    ap.add_argument("--out-json", type=Path, default=REPO / "results" / "renderer_bench.json")
    ap.add_argument("--out-png", type=Path, default=REPO / "deck_assets" / "render_check.png")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cv2.setNumThreads(2)

    scenario, source = build_scenario(args.seed)
    synthetic = source.endswith("make_test_scenario")
    cal = defaults.stereo_calibration()
    fB = cal.fx * cal.baseline_m
    sgbm = make_sgbm()
    poses = poses_for(scenario, args.n, synthetic)
    per_frame: list[dict] = []
    sheet_rows: list[np.ndarray] = []
    sheet_at = {0, args.n // 2, args.n - 1}
    agg_valid, agg_total, agg_err = 0, 0, []

    load_before = machine_cpu_load_pct()
    with ThreeRenderer(shadows=not args.no_shadows, color_transport="rgb" if args.rgb else "yuv420") as r:
        r.load_scenario(scenario)
        load_ms = r.stats()["load"]["mean"]
        r.reset_stats()
        for k, pose in enumerate(poses):
            t = k / defaults.CAMERA_HZ_BATCH
            state = {"pose": pose, "t": t, "seq": k, "dynamic": dynamic_state(scenario, t)}
            left, right = r.render_stereo(state)
            gt = r.render_gt(state)
            gray_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
            disp = sgbm.compute(gray_l, right).astype(np.float32) / 16.0
            depth, sem = gt["depth"], gt["semantic"]
            d_gt = np.where(np.isfinite(depth), fB / np.maximum(depth, 1e-6), 0.0)
            ground = np.isin(sem, GROUND_SEM) & (depth <= GROUND_RANGE_M)
            ground[:, :SGBM_NUM_DISP] = False  # no match possible left of numDisparities
            valid = ground & (disp > 0)
            err = np.abs(disp - d_gt)[valid]
            n_ground = int(ground.sum())
            frac = float(valid.sum()) / max(n_ground, 1)
            agg_valid += int(valid.sum()); agg_total += n_ground; agg_err.append(err)
            per_frame.append({"k": k, "ms": r._timings["stereo"][-1],
                              "ground_px": n_ground, "valid_frac": frac,
                              "median_abs_err_px": float(np.median(err)) if err.size else None,
                              "exposure": r.last_js.get("exposure")})
            if k in sheet_at:
                inv_d = disp <= 0
                row = [label(left, f"left RGB  k={k}"), label(cv2.cvtColor(right, cv2.COLOR_GRAY2RGB), "right GRAY"),
                       label(colorize(disp, SGBM_NUM_DISP * 0.75, inv_d), f"SGBM disp (valid ground {frac:.0%})"),
                       label(colorize(np.where(np.isfinite(depth), depth, 0), 30.0, ~np.isfinite(depth)), "GT depth 0-30 m"),
                       label(SEM_PALETTE[np.clip(sem, 0, 4)], "GT semantic (5-class)")]
                sheet_rows.append(np.concatenate([cv2.resize(x, (cal.width // 2, cal.height // 2), interpolation=cv2.INTER_AREA) for x in row], axis=1))
            if k % 10 == 0:
                log.info("frame %d: %.1f ms, valid ground %.1f%%", k, r._timings["stereo"][-1], 100 * frac)
        st = r.stats()
        load_after = machine_cpu_load_pct()
        info = dict(r.info)
        chase = r.render_chase({"pose": poses[len(poses) // 2], "t": 0.0, "dynamic": dynamic_state(scenario, 0.0)}, 1280, 720)
        events, event_rows = render_events(r, scenario, poses[len(poses) // 3], sgbm, fB)

    err_all = np.concatenate(agg_err) if agg_err else np.zeros(0)
    valid_frac = agg_valid / max(agg_total, 1)
    ms = st["stereo"]
    go = {
        "ms_per_pair_mean_le_120": ms["mean"] <= GO_MS_PER_PAIR,
        "hardware_webgl": "swiftshader" not in info["renderer"].lower(),
        "sgbm_valid_ground_ge_60pct": valid_frac >= GO_VALID_FRAC,
    }
    go["GO"] = all(go.values())
    res = {
        "what": "Three.js stereo renderer GO/NO-GO benchmark (label: Simulated)",
        "measured_with": "scripts/bench_renderer.py",
        "date_unix": time.time(),
        "scenario_source": source, "seed": args.seed, "n_pairs": args.n, "shadows": not args.no_shadows,
        "color_transport": "rgb" if args.rgb else "yuv420",
        "renderer": info["renderer"], "vendor": info["vendor"], "three_revision": info["three"],
        "image": [cal.width, cal.height],
        "machine_cpu_load_pct_before_after": [load_before, load_after],
        "note": "Wall-clock ms include Playwright transfer (CPU-bound in the Node driver / CDP pipe) and are "
                "sensitive to concurrent load on this shared laptop; stereo_ms_js_side is the browser-side render+readback+encode.",
        "scenario_load_ms": load_ms,
        "stereo_ms_per_pair_incl_decode": ms,
        "stereo_ms_js_side": st["stereo_js"],
        "gt_ms_per_frame": st.get("gt"),
        "sgbm": {"mode": "SGBM_3WAY", "num_disparities": SGBM_NUM_DISP, "block_size": SGBM_BLOCK,
                 "ground_def": f"GT semantic in {GROUND_SEM} ({[SEM_CLASSES[i] for i in GROUND_SEM]}), GT depth <= {GROUND_RANGE_M} m, u >= {SGBM_NUM_DISP}",
                 "valid_ground_fraction": valid_frac,
                 "median_abs_disp_err_px": float(np.median(err_all)) if err_all.size else None,
                 "mean_abs_disp_err_px": float(np.mean(err_all)) if err_all.size else None,
                 "bad_gt_1px_fraction": float(np.mean(err_all > 1.0)) if err_all.size else None,
                 "bad_gt_3px_fraction": float(np.mean(err_all > 3.0)) if err_all.size else None},
        "go_criteria": go,
        "lighting_events": events,
        "per_frame": per_frame,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(args.out_png), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_png.with_name("render_chase.png")), cv2.cvtColor(chase, cv2.COLOR_RGB2BGR))
    if event_rows:
        cv2.imwrite(str(args.out_png.with_name("render_events.png")), cv2.cvtColor(np.concatenate(event_rows, axis=0), cv2.COLOR_RGB2BGR))
    log.info("stereo %.1f ms mean / %.1f ms p95 (js %.1f ms) | SGBM valid ground %.1f%%, median |err| %.3f px | GO=%s",
             ms["mean"], ms["p95"], st["stereo_js"]["mean"], 100 * valid_frac, res["sgbm"]["median_abs_disp_err_px"] or -1, go["GO"])
    return 0 if go["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
