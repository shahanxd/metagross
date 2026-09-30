"""DEV-only perception scoring harness: the onboard :class:`Perception` against ground truth.

This is **evaluation** code: it may read scenario files, the GT hazard raster and GT poses.
It never runs on EVAL seeds (0-59); :func:`load_dev_scenario` refuses anything outside
``DEV_SEEDS`` (100-129).

Frame sources
-------------
* ``tier0``  : the simulator's Tier-0 synthetic disparity sensor
  (:class:`metagross.sim.sensors.Tier0DepthSensor`) at GT poses placed along the straight
  start->goal line of a DEV scenario (vehicle settled on the terrain by the simulator).
* ``stereo`` : Three.js rendered stereo pairs (:class:`metagross.sim.render.bridge.ThreeRenderer`)
  at the same kind of GT poses, run through SGBM inside ``Perception``. Rendered frames are cached
  in ``results/raw/perception/stereo_<seed>_<kind>.npz`` so before/after runs score identical frames.

Sequences
---------
* ``drive``    : body origin at ``s`` = F1_DRIVE_M[0] .. F1_DRIVE_M[1] along the start->goal line
  (off-hazard false-positive rates on rolling terrain).
* ``approach`` : straight approach to a ditch crossing of the start->goal line, body origin from
  ``APPROACH_D0_M`` to ``APPROACH_D1_M`` before the crossing, ``STEP_M`` apart (1 m/s at 5 Hz).

Metrics (per sequence and pooled per mode; BEV cells follow the egocentric convention of
:mod:`metagross.autonomy.perception.bev`, GT is looked up at the planar GT pose x, y, yaw):

* ``cand_rate_off``    : DITCH_CANDIDATE share of observed cells farther than
  ``OFF_HAZARD_BUFFER_M`` from any GT hazard (lethal | water | dynamic rest footprint).
* ``lethal_rate_off``  : POSITIVE | DEPRESSION | DITCH_CANDIDATE share of the same cells.
* per-ditch ``detection``: the first (farthest) range from the camera to the ditch's near lip
  on the path corridor (|y| <= vehicle half width) at which >= ``DETECT_FRACTION`` of its path
  cells are DITCH_CANDIDATE / DEPRESSION / POSITIVE (``r_first_m``), and the farthest range from
  which every nearer frame is detected (``r_stable_m``); compared with the Matthies-Rankin
  theory range of that width (:func:`metagross.eval.theory.ditch_detection_range_m`).
* ``certified_hazard_in_envelope``: GT lethal cells in the path corridor within the stopping
  envelope ``L/2 + v^2/2a + v T_r + B`` (v = platform cap) that perception certifies as ground
  (``certified_local`` if the output has it, else ``state == GROUND``), summed over frames.
* ``certified_hazard_any``: the same for all GT lethal cells in the BEV grid.
* ``timing_ms``        : p50 / p95 of wall-clock ``process()`` time per frame (SGBM included in
  stereo mode; the segmenter runs at 1/3 rate as in the node when ``--semantics``).

Run (PowerShell)::

    $env:OMP_NUM_THREADS=2; .venv\\Scripts\\python.exe -m metagross.eval.perception_dev --section before --modes tier0 stereo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import CellState, SensorFrame

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCEN_DIR = REPO_ROOT / "data" / "scenarios" / "dev"
RESULTS_JSON = REPO_ROOT / "results" / "perception_dev.json"
RAW_DIR = REPO_ROOT / "results" / "raw" / "perception"

DEV_SEEDS = tuple(range(100, 130))  # never EVAL seeds 0-59
STEP_M = 0.2  # frame spacing along a sequence (1 m/s at CAMERA_HZ_BATCH = 5 Hz)
F1_DRIVE_M = (2.0, 22.0)  # body-origin arc length along start->goal for 'drive' sequences (m)
DRIVE_STEP_M = 0.5  # drive frames are spaced wider (false-positive statistics, not ranges)
APPROACH_D0_M = 13.0  # approach starts this far (m) before the ditch crossing (beyond the 12 m range cap)
APPROACH_D1_M = 1.6  # ... and ends this close (near-lip blind zone of the camera ~1.2 m)
OFF_HAZARD_BUFFER_M = 0.5  # cells this close to a GT hazard are ambiguous (lip rasterisation) and not scored off-hazard
DETECT_FRACTION = 0.5  # a ditch is detected in a frame when >= this share of its path cells are flagged
MIN_PATH_CELLS = 3  # frames with fewer GT path cells of the ditch in the grid are not scored
DETECT_MIN_RANGE_M = 2.0  # near-lip ranges below this (camera blind zone ~1.3 m + lip) are not scored for detection
SUBSAMPLE_OFFS_M = (-0.025, 0.025)  # 2x2 sub-samples per 0.1 m BEV cell for the GT lookup
SCENE_T0_S = 20.0  # scene time of the first frame (no DEV lighting event at the scored seeds)
FLAGGED_STATES = (CellState.DITCH_CANDIDATE, CellState.DEPRESSION, CellState.POSITIVE)
LETHAL_STATES = (CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE)
OBSERVED_STATES = (CellState.GROUND, CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE,
                   CellState.WATER, CellState.DYNAMIC)

# Default DEV sequence plan (families by seed % 6: 102/108/114 F1, 103/109/115 F2, 104/110 F3).
TIER0_PLAN = {"drive": (102, 108, 114, 104), "approach": (103, 109, 115, 110)}
STEREO_PLAN = {"drive": (102,), "approach": (103,)}


# =========================================================================== scenarios / GT
def load_dev_scenario(seed: int) -> dict:
    """Scenario dict of a DEV seed; refuses EVAL seeds."""
    if seed not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is not a DEV seed (100-129)")
    return json.loads((SCEN_DIR / f"{seed}.json").read_text())


def line_dir(sc: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(start xy, unit direction, length) of the straight start->goal line (world, m)."""
    a = np.asarray(sc["start"]["xy"], float)
    b = np.asarray(sc["goal"]["xy"], float)
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / L, L


@dataclass
class Crossing:
    """Where the start->goal line crosses a ditch centre polyline outside its bypass gaps."""

    s_m: float  # arc length along the start->goal line (m)
    hazard: int  # index into scenario['hazards']
    width_m: float
    depth_m: float
    angle_deg: float  # 90 = perpendicular


def ditch_crossings(sc: dict, gap_margin_m: float = 2.0) -> list[Crossing]:
    """Crossings of the start->goal line with ditch polylines (sorted by arc length)."""
    a, u, L = line_dir(sc)
    out: list[Crossing] = []
    for k, hz in enumerate(sc.get("hazards", [])):
        if hz.get("type") != "ditch":
            continue
        P = np.asarray(hz["polyline"], float)
        seg = np.diff(P, axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        for i, d in enumerate(seg):
            M = np.array([[u[0], -d[0]], [u[1], -d[1]]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            s, r = np.linalg.solve(M, P[i] - a)
            if not (0.0 <= r <= 1.0 and 0.0 <= s <= L):
                continue
            s_arc = arc[i] + r * float(np.linalg.norm(d))
            if any(g0 - gap_margin_m <= s_arc <= g1 + gap_margin_m for g0, g1 in hz.get("gaps", [])):
                continue
            cosang = abs(float(np.dot(u, d / np.linalg.norm(d))))
            out.append(Crossing(float(s), k, float(hz["width"]), float(hz["depth"]), math.degrees(math.acos(min(cosang, 1.0)))))
    return sorted(out, key=lambda c: c.s_m)


class ScenarioGT:
    """Simulator world (read-only) + GT rasters of one DEV scenario."""

    def __init__(self, seed: int) -> None:
        from scipy.ndimage import distance_transform_edt

        from metagross.sim.hazards import ditch_drop, DITCH_MASK_MIN_DROP_M
        from metagross.sim.world import World

        self.seed = seed
        self.sc = load_dev_scenario(seed)
        self.world = World(self.sc, sensor_mode="tier0")
        hz = self.world.hazards
        self.grid = hz.grid
        self.lethal = hz.lethal
        any_hazard = hz.hazard
        self.dist_m = distance_transform_edt(~any_hazard) * self.grid.res
        self.ditch_id = np.full(self.grid.shape, -1, np.int16)
        for k, h in enumerate(self.sc.get("hazards", [])):
            if h.get("type") == "ditch":
                self.ditch_id[ditch_drop(self.grid, h) > DITCH_MASK_MIN_DROP_M] = k
        self.a, self.u, self.L = line_dir(self.sc)
        self.yaw = math.atan2(self.u[1], self.u[0])

    def pose_at(self, s_m: float) -> list[float]:
        """Settle the vehicle at arc length ``s_m`` of the start->goal line; GT pose6."""
        xy = self.a + s_m * self.u
        self.world.vehicle.reset(float(xy[0]), float(xy[1]), self.yaw)
        return [float(v) for v in self.world.state.pose6()]

    def cell_gt(self, pose6: list[float], X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
        """GT per BEV cell (body X, Y centres): lethal (any sub-sample), ditch id, off-hazard."""
        c, s = math.cos(pose6[5]), math.sin(pose6[5])
        lethal = np.zeros(X.shape, bool)
        obj = np.zeros(X.shape, bool)
        water = np.zeros(X.shape, bool)
        did = np.full(X.shape, -1, np.int16)
        dist = np.full(X.shape, np.inf)
        hz = self.world.hazards
        for dx in SUBSAMPLE_OFFS_M:
            for dy in SUBSAMPLE_OFFS_M:
                wx = pose6[0] + c * (X + dx) - s * (Y + dy)
                wy = pose6[1] + s * (X + dx) + c * (Y + dy)
                inside = self.grid.contains(wx, wy)
                i, j = self.grid.xy_to_ij(wx, wy)
                lethal |= inside & self.lethal[i, j]
                obj |= inside & hz.object[i, j]
                water |= inside & hz.water[i, j]
                did = np.where(inside & (self.ditch_id[i, j] >= 0), self.ditch_id[i, j], did)
                dist = np.minimum(dist, np.where(inside, self.dist_m[i, j], 0.0))
        return {"lethal": lethal, "ditch_id": did, "off_hazard": dist > OFF_HAZARD_BUFFER_M, "object": obj, "water": water}


# =========================================================================== sequences
@dataclass
class Sequence:
    """A list of GT poses along the start->goal line of one DEV seed."""

    seed: int
    kind: str  # 'drive' | 'approach'
    s_values: np.ndarray  # body-origin arc lengths (m)
    crossing: Optional[Crossing] = None

    @property
    def name(self) -> str:
        return f"{self.seed}_{self.kind}"


def make_sequence(gt: ScenarioGT, kind: str, step_m: Optional[float] = None) -> Optional[Sequence]:
    """Drive along the line, or approach its first ditch crossing (None if there is none)."""
    if kind == "drive":
        st = step_m or DRIVE_STEP_M
        s = np.arange(F1_DRIVE_M[0], min(F1_DRIVE_M[1], gt.L - 2.0) + 1e-9, st)
        return Sequence(gt.seed, kind, s)
    cr = ditch_crossings(gt.sc)
    if not cr:
        return None
    c0 = cr[0]
    st = step_m or STEP_M
    s = c0.s_m - np.arange(APPROACH_D0_M, APPROACH_D1_M - 1e-9, -st)
    return Sequence(gt.seed, kind, s[s > 0.5], c0)


def tier0_frames(gt: ScenarioGT, seq: Sequence) -> Iterable[tuple[SensorFrame, list[float]]]:
    """Tier-0 synthetic disparity frames at the sequence poses (deterministic noise)."""
    w = gt.world
    for k, s in enumerate(seq.s_values):
        pose = gt.pose_at(float(s))
        t = SCENE_T0_S + k / defaults.CAMERA_HZ_BATCH
        rng = np.random.default_rng([seq.seed, k, 0x7E0])
        res = w.depth_sensor.render(w.camera_pose(), w.dynamic_prims(t), w.lighting_state(t), rng)
        yield SensorFrame(t=t, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0,
                          gyro_z_rps=0.0, sensor_mode="tier0_disparity", disparity=res.disparity), pose


def stereo_cache_path(seq: Sequence) -> Path:
    return RAW_DIR / f"stereo_{seq.name}.npz"


def render_stereo_sequence(gt: ScenarioGT, seq: Sequence) -> Path:
    """Render and cache the stereo pairs of a sequence (Chrome; AE settled at the first pose)."""
    from metagross.sim.render.bridge import ThreeRenderer

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r = ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE, shadows=True, color_transport="rgb")
    lefts, rights, poses, ts = [], [], [], []
    try:
        r.load_scenario(gt.sc)
        for k, s in enumerate(seq.s_values):
            pose = gt.pose_at(float(s))
            t = SCENE_T0_S + k / defaults.CAMERA_HZ_BATCH
            st = gt.world.render_state()
            st.update(t=t, lighting=gt.world.lighting_state(t), seq=k)
            if k == 0:
                for j, dt in enumerate((-12.0, -6.0, 0.0)):  # auto-exposure settle (> 5 s apart)
                    left, right = r.render_stereo(dict(st, t=t + dt, seq=j))
            else:
                left, right = r.render_stereo(st)
            lefts.append(left.copy()), rights.append(right.copy()), poses.append(pose), ts.append(t)
    finally:
        r.close()
    p = stereo_cache_path(seq)
    np.savez_compressed(p, left=np.stack(lefts), right=np.stack(rights), pose=np.asarray(poses), t=np.asarray(ts),
                        s=np.asarray(seq.s_values))
    LOG.info("rendered %d stereo frames -> %s", len(lefts), p)
    return p


def stereo_frames(gt: ScenarioGT, seq: Sequence) -> Iterable[tuple[SensorFrame, list[float]]]:
    p = stereo_cache_path(seq)
    if not p.exists():
        render_stereo_sequence(gt, seq)
    z = np.load(p)
    for k in range(z["left"].shape[0]):
        yield SensorFrame(t=float(z["t"][k]), seq=k, left_rgb=z["left"][k], right_gray=z["right"][k], wheel_angle_l_rad=0.0,
                          wheel_angle_r_rad=0.0, gyro_z_rps=0.0, sensor_mode="stereo"), [float(v) for v in z["pose"][k]]


# =========================================================================== scoring
def stopping_envelope_m(v_mps: float = defaults.VEHICLE.max_speed_mps) -> float:
    """Body-x extent (m) of the stopping envelope: front overhang + v^2/2a + v T_r + B."""
    from metagross.eval.theory import T_REACTION_S, stopping_distance_m

    d = float(stopping_distance_m(v_mps, defaults.BRAKE_DECEL_MPS2, T_REACTION_S, defaults.GOVERNOR_MARGIN_M))
    return 0.5 * defaults.VEHICLE.length_m + d


def theory_r_det_m(width_m: float) -> float:
    """Matthies-Rankin first-detection range (m, from the camera) of a ditch of this width."""
    from metagross.eval.theory import ditch_detection_range_m

    return float(ditch_detection_range_m(width_m, defaults.CAM_HEIGHT_M, defaults.FX, float(defaults.MIN_PIXELS_ON_TARGET)))


@dataclass
class FrameScore:
    n_obs_off: int
    n_cand_off: int
    n_lethal_off: int
    n_pos_off: int
    haz_env: int  # GT lethal cells inside the stopping envelope corridor
    haz_env_cert: int  # ... certified as ground
    haz_any: int  # GT lethal cells in the grid (observed or not)
    haz_any_cert: int
    haz_env_cert_ditch: int  # certified envelope hazard cells that are GT ditch cells
    ditch_path_cert: int  # GT path cells of the tracked ditch certified as ground (any range)
    ditch_path: int  # GT path cells of the tracked ditch in the grid
    ditch_flagged: int
    ditch_lip_range_m: float  # range from the camera to the nearest GT path cell of the ditch (m)
    ms: float
    stages: dict[str, float] = field(default_factory=dict)


def score_frame(out: dict[str, Any], gt_cells: dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray,
                ditch_k: Optional[int], ms: float, envelope_m: float) -> FrameScore:
    st = out["cell_state_local"]
    cert = out.get("certified_local")
    if cert is None:
        cert = st == CellState.GROUND
    obs = np.isin(st, np.array(OBSERVED_STATES, np.uint8))
    off = gt_cells["off_hazard"] & obs
    cand = st == CellState.DITCH_CANDIDATE
    lethal_st = np.isin(st, np.array(LETHAL_STATES, np.uint8))
    corridor = (np.abs(Y) <= 0.5 * defaults.VEHICLE.width_m) & (X > 0.0)
    env = corridor & (X <= envelope_m)
    haz = gt_cells["lethal"]
    ditch_path = np.zeros_like(haz)
    if ditch_k is not None:
        ditch_path = corridor & (gt_cells["ditch_id"] == ditch_k)
    flagged = np.isin(st, np.array(FLAGGED_STATES, np.uint8))
    lip = float(X[ditch_path].min() - defaults.CAM_FORWARD_M) if ditch_path.any() else float("nan")
    return FrameScore(int(off.sum()), int((off & cand).sum()), int((off & lethal_st).sum()),
                      int((off & (st == CellState.POSITIVE)).sum()), int((env & haz).sum()), int((env & haz & cert).sum()),
                      int(haz.sum()), int((haz & cert).sum()), int((env & haz & cert & (gt_cells["ditch_id"] >= 0)).sum()),
                      int((ditch_path & cert).sum()), int(ditch_path.sum()), int((ditch_path & flagged).sum()), lip,
                      ms, {k: float(v) for k, v in (out.get("timings_ms") or {}).items()})


def detection_ranges(scores: list[FrameScore]) -> dict[str, float]:
    """r_first / r_stable (m from the camera to the near lip) of an approach sequence."""
    rows = [(f.ditch_lip_range_m, f.ditch_flagged / f.ditch_path) for f in scores
            if f.ditch_path >= MIN_PATH_CELLS and np.isfinite(f.ditch_lip_range_m) and f.ditch_lip_range_m >= DETECT_MIN_RANGE_M]
    if not rows:
        return {"r_first_m": float("nan"), "r_stable_m": float("nan"), "n_frames": 0}
    rows.sort(key=lambda r: -r[0])  # far -> near
    det = [r for r, fr in rows if fr >= DETECT_FRACTION]
    r_first = max(det) if det else 0.0
    r_stable = 0.0
    for r, fr in reversed(rows):  # near -> far
        if fr < DETECT_FRACTION:
            break
        r_stable = r
    return {"r_first_m": round(r_first, 2), "r_stable_m": round(r_stable, 2), "n_frames": len(rows),
            "max_range_scored_m": round(rows[0][0], 2)}


def _pct(a: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(a, float), q)) if a else float("nan")


def summarise(scores: list[FrameScore]) -> dict[str, Any]:
    n_obs = sum(f.n_obs_off for f in scores)
    ms = [f.ms for f in scores]
    stage_keys = sorted({k for f in scores for k in f.stages})
    return {
        "n_frames": len(scores),
        "cand_rate_off": round(sum(f.n_cand_off for f in scores) / max(n_obs, 1), 5),
        "lethal_rate_off": round(sum(f.n_lethal_off for f in scores) / max(n_obs, 1), 5),
        "positive_rate_off": round(sum(f.n_pos_off for f in scores) / max(n_obs, 1), 5),
        "n_observed_off_cells": n_obs,
        "certified_hazard_in_envelope": {"cells": sum(f.haz_env_cert for f in scores), "of": sum(f.haz_env for f in scores)},
        "certified_hazard_in_envelope_ditch_cells": sum(f.haz_env_cert_ditch for f in scores),
        "certified_hazard_any": {"cells": sum(f.haz_any_cert for f in scores), "of": sum(f.haz_any for f in scores)},
        "ditch_path_certified": {"cells": sum(f.ditch_path_cert for f in scores), "of": sum(f.ditch_path for f in scores)},
        "timing_ms": {"p50": round(_pct(ms, 50), 1), "p95": round(_pct(ms, 95), 1),
                      "stages_p50": {k: round(_pct([f.stages.get(k, 0.0) for f in scores], 50), 2) for k in stage_keys}},
    }


PERCEPTION_MODULE = "metagross.autonomy.perception.pipeline"  # --perception-module swaps it (frozen baselines)


def make_perception(semantics: bool) -> Any:
    """Onboard Perception as the node builds it (segmenter at 1/3 rate when ``semantics``)."""
    import importlib

    Perception = importlib.import_module(PERCEPTION_MODULE).Perception

    seg = None
    if semantics:
        from metagross.autonomy.node import SEG_EVERY_N, EveryNthSegmenter
        from metagross.autonomy.perception.semantics import Segmenter

        seg = EveryNthSegmenter(Segmenter.from_config({"seg_threads": 2}), SEG_EVERY_N)
    return Perception(defaults.stereo_calibration(), defaults.VEHICLE, {}, segmenter=seg)


def run_sequence(gt: ScenarioGT, seq: Sequence, mode: str, semantics: bool) -> dict[str, Any]:
    """Run a fresh onboard Perception over a sequence and score every frame."""
    per = make_perception(semantics and mode == "stereo")
    X, Y = per.spec.centres()
    env = stopping_envelope_m()
    frames = tier0_frames(gt, seq) if mode == "tier0" else stereo_frames(gt, seq)
    ditch_k = seq.crossing.hazard if seq.crossing is not None else None
    scores: list[FrameScore] = []
    for frame, pose in frames:
        t0 = time.perf_counter()
        out = per.process(frame, (pose[0], pose[1], pose[5]))
        ms = (time.perf_counter() - t0) * 1e3
        scores.append(score_frame(out, gt.cell_gt(pose, X, Y), X, Y, ditch_k, ms, env))
    res = summarise(scores)
    res.update(seed=seq.seed, family=gt.sc["family"], kind=seq.kind, mode=mode)
    if seq.crossing is not None:
        det = detection_ranges(scores)
        w = seq.crossing.width_m
        det.update(width_m=round(w, 3), depth_m=round(seq.crossing.depth_m, 3), angle_deg=round(seq.crossing.angle_deg, 1),
                   theory_r_det_m=round(theory_r_det_m(w), 2))
        res["detection"] = det
    res["_scores"] = scores
    return res


def pool(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = summarise([f for r in results for f in r["_scores"]])
    return out


def evaluate(modes: list[str], semantics: bool, plans: Optional[dict[str, dict[str, tuple[int, ...]]]] = None) -> dict[str, Any]:
    """Run all DEV sequences for the given modes; per-sequence and pooled metrics."""
    import cv2

    cv2.setNumThreads(2)
    plans = plans or {"tier0": TIER0_PLAN, "stereo": STEREO_PLAN}
    section: dict[str, Any] = {}
    gts: dict[int, ScenarioGT] = {}
    for mode in modes:
        seqs: list[dict[str, Any]] = []
        for kind, seeds in plans[mode].items():
            for seed in seeds:
                gt = gts.setdefault(seed, ScenarioGT(seed))
                seq = make_sequence(gt, kind)
                if seq is None:
                    continue
                LOG.info("%s %s: %d frames", mode, seq.name, len(seq.s_values))
                seqs.append(run_sequence(gt, seq, mode, semantics))
        fam_f1 = [r for r in seqs if r["family"] == "F1_trail"]
        section[mode] = {
            "pooled": pool(seqs),
            "pooled_F1": pool(fam_f1) if fam_f1 else None,
            "sequences": [{k: v for k, v in r.items() if k != "_scores"} for r in seqs],
        }
    return section


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", required=True, help="key in results/perception_dev.json (e.g. before / after)")
    ap.add_argument("--modes", nargs="+", default=["tier0", "stereo"], choices=["tier0", "stereo"])
    ap.add_argument("--no-semantics", action="store_true", help="stereo timing without the segmenter")
    ap.add_argument("--out", default=str(RESULTS_JSON))
    ap.add_argument("--perception-module", default=PERCEPTION_MODULE,
                    help="module providing Perception (e.g. a frozen copy of an older commit on PYTHONPATH)")
    args = ap.parse_args(argv)
    globals()["PERCEPTION_MODULE"] = args.perception_module
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sec = evaluate(args.modes, semantics=not args.no_semantics)
    sec["meta"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "modes": args.modes, "semantics": not args.no_semantics,
                   "perception_module": args.perception_module,
                   "stopping_envelope_m": round(stopping_envelope_m(), 3), "off_hazard_buffer_m": OFF_HAZARD_BUFFER_M,
                   "detect_fraction": DETECT_FRACTION, "note": "DEV seeds only; wall-clock on a shared 4-core laptop"}
    p = Path(args.out)
    data = json.loads(p.read_text()) if p.exists() else {}
    data.setdefault(args.section, {}).update(sec)
    p.write_text(json.dumps(data, indent=1, default=float))
    LOG.info("wrote %s [%s]", p, args.section)
    for mode in args.modes:
        pl = sec[mode]["pooled"]
        LOG.info("%s pooled: cand_off %.4f lethal_off %.4f cert_env %s timing %s", mode, pl["cand_rate_off"],
                 pl["lethal_rate_off"], pl["certified_hazard_in_envelope"], pl["timing_ms"])
        for r in sec[mode]["sequences"]:
            LOG.info("  %s %s: cand_off %.4f lethal_off %.4f det %s", r["seed"], r["kind"], r["cand_rate_off"],
                     r["lethal_rate_off"], r.get("detection"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
