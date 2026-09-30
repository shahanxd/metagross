"""Train and test the VO integrity monitor on real KITTI frames with synthetic degradations.

Two steps (so the expensive VO passes can run in parallel processes):

``collect``  run the stereo VO over a KITTI sequence whose frames are degraded in
             random segments (:func:`metagross.eval.degrade.make_schedule`,
             deterministic seed), and store per-frame raw features, VO output, GT
             relative pose and label in ``results/raw/integrity_<seq>.npz``.
``fit``      train a logistic regression on the train sequence(s), test on a
             *different* sequence, write ``models/integrity.json`` (loaded by the
             onboard :class:`IntegrityMonitor`), ``results/integrity.json`` and
             ``deck_assets/integrity_roc.png``; replay the onboard gating logic
             on the test sequence to measure drift with vs without gating.

Label: a frame is a VO failure if VO returned no pose, or if its relative pose
error against GT exceeds 0.10 m or 1 deg. The classifier is trained and scored on
all frames after the first; the AUROC restricted to frames where VO *did* return a
pose (silent failures) is reported separately when that subset has positives.

Usage::

    python -m metagross.eval.integrity_train collect --seq 05 --max-frames 1500
    python -m metagross.eval.integrity_train collect --seq 07
    python -m metagross.eval.integrity_train fit --train 05 --test 07
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from metagross.autonomy.localization.health import (
    FEATURE_NAMES,
    HealthConfig,
    IntegrityMonitor,
    LogisticIntegrityModel,
    transform_features,
)
from metagross.autonomy.localization.vo import rotation_angle
from metagross.eval.degrade import Degrader, depth_from_sgbm, make_schedule
from metagross.eval.kitti_vo import KittiSequence, run_vo
from metagross.eval.traj_metrics import evaluate

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "results" / "raw"
RESULTS = REPO_ROOT / "results"
DECK = REPO_ROOT / "deck_assets"
MODEL_PATH = REPO_ROOT / "models" / "integrity.json"

FAIL_TRANS_M = 0.10  # per-frame relative translation error defining a VO failure
FAIL_ROT_DEG = 1.0  # per-frame relative rotation error defining a VO failure
SCHEDULE_SEED = 1234  # degradation schedule seed (offset by sequence id)
# Secondary "degraded accuracy" label (reported alongside, never used for training):
# ~2x the 95th percentile of clean-frame error observed on KITTI 07.
STRICT_TRANS_M = 0.05
STRICT_ROT_DEG = 0.25


def _npz_path(seq: str) -> Path:
    return RAW / f"integrity_{seq}.npz"


def collect(seq_id: str, max_frames: Optional[int] = None, threads: int = 2, seed: int = SCHEDULE_SEED) -> Path:
    """Degraded VO run over one sequence -> per-frame features / errors / labels (npz)."""
    cv2.setNumThreads(threads)
    seq = KittiSequence(seq_id)
    n_avail = len(seq.frames()) if seq.available() else seq.n_contiguous_frames()
    if n_avail < 2:
        raise FileNotFoundError(f"KITTI {seq_id} not downloaded")
    if not seq.available():
        LOG.warning("sequence %s only partially downloaded: using the first %d frames", seq_id, n_avail)
    K, baseline = seq.calib()
    n = n_avail if max_frames is None else min(max_frames, n_avail)
    schedule = make_schedule(n, seed + int(seq_id))
    schedule[0] = None  # the VO init frame stays clean
    degraders: dict[int, Degrader] = {}

    def transform(k: int, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        item = schedule[k]
        if item is None:
            return left, right
        deg, seg_seed = item
        if seg_seed not in degraders:
            degraders[seg_seed] = Degrader(deg, left.shape, seg_seed)
        depth = depth_from_sgbm(left, right, K[0, 0], baseline) if deg.kind == "haze" else None
        return degraders[seg_seed](left, right, depth, frame=k)

    t0 = time.perf_counter()
    run = run_vo(seq, max_frames=n, transform=transform, with_features=True, log_every=250)
    gt = seq.gt()[:n]
    terr = np.full(n, np.nan)
    rerr = np.full(n, np.nan)
    for k in range(1, n):
        if run.ok[k]:
            T_gt = np.linalg.inv(gt[k - 1]) @ gt[k]
            E = np.linalg.inv(run.rel_vo[k]) @ T_gt
            terr[k] = float(np.linalg.norm(E[:3, 3]))
            rerr[k] = math.degrees(rotation_angle(E[:3, :3]))
    label = (~run.ok) | (terr > FAIL_TRANS_M) | (rerr > FAIL_ROT_DEG)
    label[0] = False
    X = np.array([[f[name] for name in FEATURE_NAMES] for f in run.features])
    kinds = np.array(["clean" if s is None else s[0].kind for s in schedule])
    sev = np.array([0 if s is None else s[0].severity for s in schedule])
    RAW.mkdir(parents=True, exist_ok=True)
    out = _npz_path(seq_id)
    np.savez_compressed(out, X=X, ok=run.ok, label=label, terr=terr, rerr=rerr, rel_vo=run.rel_vo, gt=gt,
                        kinds=kinds, severity=sev, vo_ms=run.vo_ms, seed=seed + int(seq_id))
    LOG.info("collected %s: %d frames in %.0f s, degraded %.0f %%, failures %.1f %% (silent %d)", seq_id, n,
             time.perf_counter() - t0, 100 * np.mean(kinds != "clean"), 100 * label[1:].mean(),
             int((label & run.ok).sum()))
    return out


def _transform_matrix(X: np.ndarray) -> np.ndarray:
    return np.stack([transform_features(dict(zip(FEATURE_NAMES, row))) for row in X])


def integrate(rel_vo: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Chain relative poses where ``use`` is True, bridging others with the last used motion."""
    n = rel_vo.shape[0]
    poses = np.tile(np.eye(4), (n, 1, 1))
    last = np.eye(4)
    for k in range(1, n):
        if use[k]:
            last = rel_vo[k]
        poses[k] = poses[k - 1] @ last
    return poses


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def fit(train: list[str], test: str, C: float = 1.0, tag: str = "") -> dict:
    """Train on ``train`` sequences, evaluate on ``test``; writes model, results and ROC plot.

    A non-empty ``tag`` marks a secondary split: outputs get a ``_<tag>`` suffix and
    ``models/integrity.json`` is left untouched.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler

    tr = [np.load(_npz_path(s), allow_pickle=False) for s in train]
    te = np.load(_npz_path(test), allow_pickle=False)
    if test in train:
        raise ValueError("test sequence must differ from the training sequences")

    def domain(d):  # every frame after the VO init frame
        m = np.ones(d["ok"].shape[0], bool)
        m[0] = False
        return m

    Xtr = np.concatenate([_transform_matrix(d["X"][domain(d)]) for d in tr])
    ytr = np.concatenate([d["label"][domain(d)] for d in tr]).astype(int)
    m_te = domain(te)
    Xte, yte = _transform_matrix(te["X"][m_te]), te["label"][m_te].astype(int)
    ok_te = te["ok"][m_te].astype(bool)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(Xtr), ytr)
    model = LogisticIntegrityModel(scaler.mean_, scaler.scale_, clf.coef_.ravel(), float(clf.intercept_[0]),
                                   source="trained")
    fallback = LogisticIntegrityModel.fallback()

    def p_of(m: LogisticIntegrityModel, Xt: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(m.logit(Xt), -50, 50)))

    p_tr, p_te, p_fb = p_of(model, Xtr), p_of(model, Xte), p_of(fallback, Xte)
    auc = {
        "auroc_train": float(roc_auc_score(ytr, p_tr)),
        "auroc_test": float(roc_auc_score(yte, p_te)),
        "auroc_test_fallback_model": float(roc_auc_score(yte, p_fb)),
    }
    # silent failures only: frames where VO *did* return a pose (the hard case)
    n_silent = int(yte[ok_te].sum())
    auc["silent_failures_test"] = n_silent
    auc["auroc_test_silent_only"] = (float(roc_auc_score(yte[ok_te], p_te[ok_te]))
                                     if 0 < n_silent < int(ok_te.sum()) else None)
    # Secondary (not the spec label): does p_fail rank the accuracy of frames that returned a pose?
    from scipy.stats import spearmanr

    terr_te, rerr_te = te["terr"][m_te], te["rerr"][m_te]
    valid = ok_te & np.isfinite(terr_te)
    if valid.sum() > 2 and np.ptp(terr_te[valid]) > 0 and np.ptp(p_te[valid]) > 0:
        rho = spearmanr(p_te[valid], terr_te[valid])
        auc["spearman_pfail_vs_trans_err"] = float(rho.statistic)
        auc["spearman_p_value"] = float(rho.pvalue)
    else:  # rank correlation undefined for constant input
        auc["spearman_pfail_vs_trans_err"] = auc["spearman_p_value"] = None
    strict = (terr_te[valid] > STRICT_TRANS_M) | (rerr_te[valid] > STRICT_ROT_DEG)
    auc["strict_label_positives_test"] = int(strict.sum())
    auc["auroc_test_strict_label"] = (float(roc_auc_score(strict, p_te[valid]))
                                      if 0 < strict.sum() < strict.size else None)

    # per-kind breakdown on the test sequence
    per_kind = {}
    kinds_te = te["kinds"][m_te]
    for kd in sorted(set(kinds_te.tolist())):
        s = kinds_te == kd
        entry = {"frames": int(s.sum()), "fail_rate": float(yte[s].mean()), "mean_p_fail": float(p_te[s].mean())}
        if 0 < yte[s].sum() < s.sum():
            entry["auroc"] = float(roc_auc_score(yte[s], p_te[s]))
        per_kind[kd] = entry

    # --- gating replay on the test sequence with the onboard monitor logic
    cfg = HealthConfig()
    mon = IntegrityMonitor(model=model, config=cfg)
    n = te["ok"].shape[0]
    q_gate = np.ones(n)
    for k in range(n):
        h = mon.update_features(dict(zip(FEATURE_NAMES, te["X"][k])), warmup=(k == 0))
        q_gate[k] = h["q_gate"]
    ok = te["ok"].astype(bool)
    use_nogate = ok.copy()
    use_gate = ok & (q_gate >= cfg.q_reject)
    use_oracle = ok & ~te["label"].astype(bool)
    gt = te["gt"]
    traj = {"no_gating": integrate(te["rel_vo"], use_nogate), "health_gating": integrate(te["rel_vo"], use_gate),
            "oracle_gating": integrate(te["rel_vo"], use_oracle)}
    drift = {k: _clean(evaluate(gt, v)) for k, v in traj.items()}
    gate_stats = {
        "q_reject": cfg.q_reject,
        "frames_rejected_by_health": int((ok & ~use_gate)[1:].sum()),
        "true_failures_rejected": int((ok & ~use_gate & te["label"].astype(bool)).sum()),
        "silent_failures_total": int((ok & te["label"].astype(bool)).sum()),
        "good_frames_rejected": int((ok & ~use_gate & ~te["label"].astype(bool)).sum()),
        "good_frames_total": int((ok & ~te["label"].astype(bool))[1:].sum()),
    }

    meta = {
        "trained_on": train, "tested_on": test, "label": f"VO failure: no pose, or rel. error > {FAIL_TRANS_M} m "
        f"or > {FAIL_ROT_DEG} deg per frame", "train_frames": int(ytr.size), "train_fail_rate": float(ytr.mean()),
        "test_frames": int(yte.size), "test_fail_rate": float(yte.mean()), **auc,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "C": C,
    }
    suffix = f"_{tag}" if tag else ""
    if not tag:  # only the primary split feeds the onboard model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(model.to_json(meta), indent=2), encoding="utf-8")

    res = {"label": "Tested", "meta": meta, "per_kind_test": per_kind, "gating": gate_stats, "drift_test": drift,
           "coef": dict(zip(FEATURE_NAMES, clf.coef_.ravel().tolist())),
           "note": "Real KITTI frames with synthetic degradations (metagross.eval.degrade). Drift: camera-only VO, "
                   "rejected frames bridged with constant velocity."}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"integrity{suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _plot_roc(yte, p_te, p_fb, auc, test, train, suffix)
    p_frames = np.concatenate([[0.0], p_te])  # frame 0 (VO init) is outside the scored domain
    _plot_timeline(p_frames, 1.0 - q_gate, te["label"].astype(bool), te["kinds"], cfg.q_reject, test, suffix)
    LOG.info("integrity: %s", {**auc, **gate_stats})
    return res


def _plot_timeline(p_fail: np.ndarray, one_minus_q: np.ndarray, label: np.ndarray, kinds: np.ndarray,
                   q_reject: float, test: str, suffix: str = "") -> None:
    """Per-frame p_fail on the degraded test sequence with degraded segments shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 2.8), dpi=150)
    k = np.arange(p_fail.size)
    start = None
    for i in range(k.size + 1):
        bad = i < k.size and kinds[i] != "clean"
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            ax.axvspan(start, i, color="#f3e3c3", lw=0)
            ax.text((start + i) / 2, 1.02, kinds[start].replace("_", " "), fontsize=5, ha="center", va="bottom",
                    rotation=90, color="#8a6d3b")
            start = None
    ax.plot(k, p_fail, color="#1f5fbf", lw=0.8, label="p_fail (instantaneous)")
    ax.plot(k, one_minus_q, color="#222222", lw=0.8, alpha=0.7, label="1 - q_gate (used for gating)")
    ax.axhline(1.0 - q_reject, color="#c0392b", lw=0.8, ls="--", label=f"reject VO above {1 - q_reject:.1f}")
    ax.plot(k[label], np.full(label.sum(), -0.05), "|", color="#c0392b", ms=6, label="true VO failure")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlim(0, k.size)
    ax.set_xlabel("frame")
    ax.set_title(f"Integrity monitor on degraded KITTI {test} (shaded = synthetic degradation)", fontsize=9,
                 loc="left", pad=48)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper right", ncol=4, bbox_to_anchor=(1.0, -0.25))
    fig.tight_layout()
    fig.savefig(DECK / f"integrity_timeline{suffix}.png", facecolor="white")
    plt.close(fig)


def _plot_roc(y: np.ndarray, p: np.ndarray, p_fb: np.ndarray, auc: dict, test: str, train: list[str],
              suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    for pp, name, col, a in ((p, "learned logistic", "#1f5fbf", auc["auroc_test"]),
                             (p_fb, "hand-set prior", "#9aa3ad", auc["auroc_test_fallback_model"])):
        fpr, tpr, _ = roc_curve(y, pp)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} (AUROC {a:.3f})")
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, ls="--")
    ax.set_xlabel("False-alarm rate (healthy VO frames flagged)")
    ax.set_ylabel("Detection rate (VO failures flagged)")
    ax.set_title(f"VO integrity monitor - KITTI {test} (trained on {', '.join(train)})", fontsize=9, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    DECK.mkdir(parents=True, exist_ok=True)
    fig.savefig(DECK / f"integrity_roc{suffix}.png", facecolor="white")
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--seq", required=True)
    c.add_argument("--max-frames", type=int, default=None)
    c.add_argument("--threads", type=int, default=2)
    f = sub.add_parser("fit")
    f.add_argument("--train", nargs="+", required=True)
    f.add_argument("--test", required=True)
    f.add_argument("--C", type=float, default=1.0)
    f.add_argument("--tag", default="", help="secondary split: suffix outputs, keep models/integrity.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "collect":
        collect(f"{int(args.seq):02d}", args.max_frames, args.threads)
    else:
        fit([f"{int(s):02d}" for s in args.train], f"{int(args.test):02d}", args.C, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
