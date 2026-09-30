"""Unit tests of the rendered-drive integrity / VO evaluation helpers (synthetic, no renderer)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from metagross.config import defaults
from metagross.eval import integrity_sim as S


def _T(x: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Camera pose: translation (x, 0, z) and rotation about the camera y axis."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[:3, 3] = [x, 0.0, z]
    return T


def test_pure_pursuit_progress_is_projected_and_monotone():
    pp = S.PurePursuit(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]), speed=1.0)
    assert pp.progress(np.array([3.0, 0.5])) == pytest.approx(3.0)
    assert pp.progress(np.array([10.2, 4.0])) == pytest.approx(14.0)
    # progress never goes back to the first segment
    assert pp.progress(np.array([5.0, 0.0])) >= 10.0


def test_pure_pursuit_steers_towards_path():
    pp = S.PurePursuit(np.array([[0.0, 0.0], [20.0, 0.0]]), speed=1.5)
    v, om = pp.command(0.0, 1.0, 0.0)  # left of the path, heading along it -> turn right
    assert v == 1.5 and om < 0
    v, om = pp.command(2.0, 0.0, math.pi)  # facing backwards -> turn in place
    assert v == 0.0 and abs(om) == pytest.approx(S.MAX_YAW_RATE_RPS)


def test_wheel_rates_inverse_kinematics():
    wl, wr = S.wheel_rates(1.0, 0.0)
    assert wl == pytest.approx(wr) == pytest.approx(1.0 / defaults.VEHICLE.wheel_radius_m)
    wl, wr = S.wheel_rates(0.0, 0.5)
    assert wl == pytest.approx(-wr) and wr > 0


def test_drive_speed_stretches_f5_over_lighting_events():
    assert S.drive_speed(102) == S.SPEEDS_MPS[102 % 3]
    sc = {"lighting": {"events": [{"t0": 10.0, "t1": 49.0}]}}
    v = S.drive_speed(104, sc, path_len_m=40.0)
    assert v == pytest.approx(40.0 / (49.0 + S.EVENT_TAIL_S))
    assert S.drive_speed(104, sc, path_len_m=1.0) == S.F5_MIN_SPEED_MPS


def test_vo_labels_and_drift():
    n = 6
    rel_gt = np.stack([np.eye(4)] + [_T(z=0.3) for _ in range(n - 1)])
    rel_vo = rel_gt.copy()
    rel_vo[2] = _T(z=0.45)  # 0.15 m error -> failure
    rel_vo[4] = np.nan  # VO failed
    rel_vo[5] = _T(z=0.3, yaw_deg=2.0)  # 2 deg error -> failure
    label, terr, _ = S.vo_labels(rel_vo, rel_gt)
    assert label.tolist() == [False, False, True, False, True, True]
    assert terr[1] == pytest.approx(0.0, abs=1e-12) and terr[2] == pytest.approx(0.15)
    d = S.vo_drift_pct(rel_gt, rel_gt)
    assert d["drift_pct"] == pytest.approx(0.0, abs=1e-9) and d["path_m"] == pytest.approx(1.5)
    scaled = np.stack([np.eye(4)] + [_T(z=0.33) for _ in range(n - 1)])
    assert S.vo_drift_pct(scaled, rel_gt)["drift_pct"] == pytest.approx(10.0)


def test_segments_and_level_onsets():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
    assert S._segments(m) == [(1, 3), (4, 5), (7, 8)]
    lv = np.array([0, 1, 2, 2, 1, 0, 2, 2])
    assert S.level_entries(lv, 1).tolist() == [1, 6]
    assert S.level_entries(lv, 2).tolist() == [2, 6]


def test_supervisor_levels_use_real_hysteresis():
    t = np.arange(0, 4.0, 0.2)
    q = np.where(t < 1.0, 0.95, np.where(t < 2.0, 0.3, 0.95))
    lv = S.supervisor_levels(t, q)
    assert lv[0] == 0 and lv[int(1.2 / 0.2)] == 2
    # upgrades one level at a time after the hold time, so DEGRADED is left only after >= 1 s
    first_ok = int(np.argmax(t >= 2.0))
    assert lv[first_ok] == 2


def test_event_masks_nominal_excludes_event_tail(tmp_path: Path):
    n = 20
    t = np.arange(n) * 0.2
    light = np.array([""] * n, dtype=object)
    light[5:8] = "glare"
    meta = {"t": t, "light": light.astype(str), "dyn_triggered": np.zeros(n, bool), "dyn_range": np.full(n, np.inf),
            "dyn_bearing": np.zeros(n)}
    meta["dyn_triggered"][15:] = True
    meta["dyn_range"][15:] = 3.0
    d = S.Drive(100, "F5_lighting", 1.0, meta, tmp_path)
    ev = S.event_masks(d)
    assert ev["glare"].sum() == 3 and ev["obstacle"].sum() == 5
    tail = int(round(S.EVENT_TAIL_S / 0.2))
    assert not ev["nominal"][5:8 + tail].any() and ev["nominal"][8 + tail]
    assert ev["nominal"][:5].all()


def _synthetic_features(n: int, fail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Raw 12-feature rows: failures have few inliers and high reprojection error."""
    from metagross.autonomy.localization.health import FEATURE_NAMES

    X = np.zeros((n, len(FEATURE_NAMES)))
    X[:, 0] = np.where(fail, rng.uniform(0, 30, n), rng.uniform(150, 500, n))  # inliers
    X[:, 1] = np.where(fail, rng.uniform(0.1, 0.5, n), rng.uniform(0.9, 1.0, n))  # inlier ratio
    X[:, 2] = np.where(fail, rng.uniform(1.5, 5.0, n), rng.uniform(0.3, 0.8, n))  # reprojection rmse px
    X[:, 3] = np.where(fail, rng.uniform(0.0, 0.3, n), rng.uniform(0.6, 0.9, n))  # coverage
    X[:, 4] = rng.uniform(3, 10, n)  # track age
    X[:, 5] = np.where(fail, rng.uniform(0, 1e3, n), rng.uniform(1e5, 1e6, n))  # Hessian min eigenvalue
    X[:, 6] = rng.uniform(300, 1200, n)  # Laplacian variance
    X[:, 9] = rng.uniform(0.2, 0.3, n)  # RMS contrast
    X[:, 10] = rng.uniform(0.15, 0.3, n)  # dark channel
    X[:, 11] = rng.uniform(0.9, 1.0, n)  # disparity density
    return X


def test_platt_folds_into_model_and_keeps_ranking():
    from metagross.autonomy.localization.health import LogisticIntegrityModel

    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.1
    X = _synthetic_features(400, y, rng)
    base = LogisticIntegrityModel.fallback()
    cal = S.platt_recalibrate(base, X, y)
    a = cal.meta["platt_a"]
    assert a > 0  # a positive slope keeps the ranking, hence the AUROC
    assert S.auroc(cal, X, y) == pytest.approx(S.auroc(base, X, y))
    assert np.allclose(cal.logit(S.transformed(X)), a * base.logit(S.transformed(X)) + cal.meta["platt_b"])


def test_retrain_balances_domains_and_separates():
    rng = np.random.default_rng(1)
    ya = rng.random(300) < 0.2
    yb = rng.random(3000) < 0.02
    m = S.retrain([(_synthetic_features(300, ya, rng), ya), (_synthetic_features(3000, yb, rng), yb)])
    yt = rng.random(500) < 0.1
    assert S.auroc(m, _synthetic_features(500, yt, rng), yt) > 0.95
    assert S.auroc(m, np.zeros((5, 12)), np.zeros(5, bool)) is None


def _seed_data(fail_idx: list[int], light_idx: list[int], n: int = 60) -> "S.SeedData":
    rng = np.random.default_rng(3)
    label = np.zeros(n, bool)
    label[fail_idx] = True
    t = np.arange(n) * S.FRAME_DT_S
    light = np.array([""] * n, dtype=object)
    light[light_idx] = "glare"
    meta = {"t": t, "light": light.astype(str), "dyn_triggered": np.zeros(n, bool), "dyn_range": np.full(n, np.inf),
            "dyn_bearing": np.zeros(n)}
    events = S.event_masks(S.Drive(100, "F5_lighting", 1.0, meta, Path(".")))
    kinds = np.array(["clean"] * n, dtype=object)
    kinds[light_idx] = "sun_flare"
    return S.SeedData(100, "F5_lighting", t, _synthetic_features(n, label, rng), label, np.zeros(n), np.zeros(n),
                      events, {"degrade_kind": kinds.astype(str)}, np.tile(np.eye(4), (n, 1, 1)))


def test_score_model_counts_nominal_onsets_and_event_detection():
    from metagross.autonomy.localization.health import LogisticIntegrityModel

    d = _seed_data(fail_idx=[10, 30, 31], light_idx=[29, 30, 31, 32])
    r = S.score_model(LogisticIntegrityModel.fallback(), [d])
    # the isolated failure at frame 10 is a nominal DEGRADED onset, but not a false alarm
    assert r["nominal_degraded_onsets"] == 1 and r["nominal_degraded_false_alarms"] == 0
    assert r["events_total"] == 1 and r["events_with_vo_failure"] == 1
    assert r["events_with_vo_failure_flagged_degraded"] == 1
    assert r["auroc_vo_failure"] == pytest.approx(1.0)
    s = S.score_degraded(LogisticIntegrityModel.fallback(), [d])
    assert s["per_kind"]["sun_flare"]["segments"] == 1 and s["segments_vo_failed_flagged_caution"] == 1


def test_dev_seed_guard():
    with pytest.raises(ValueError):
        S.load_dev_scenario(5)
