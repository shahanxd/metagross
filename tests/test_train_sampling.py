"""Step-indexed stream sampler, repeat-factor sampling, deadline stop and exact resume."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from metagross.train.sampling import (  # noqa: E402
    build_stream,
    epoch_step_range,
    n_epochs_for,
    pass_indices,
    repeat_factors,
    stream_length,
)
from metagross.train.train_seg import atomic_save, parse_deadline  # noqa: E402


def test_uniform_stream_is_concatenated_permutations() -> None:
    s = build_stream(5, 13, seed=3)
    assert len(s) == 13
    assert sorted(s[:5].tolist()) == list(range(5)) and sorted(s[5:10].tolist()) == list(range(5))
    assert np.array_equal(s, build_stream(5, 13, seed=3))  # deterministic
    assert not np.array_equal(s, build_stream(5, 13, seed=4))
    assert np.array_equal(build_stream(5, 7, seed=3), s[:7])  # prefix-stable: resume sees the same batches


def test_repeat_factors_boost_rare_class_only() -> None:
    fr = np.zeros((10, 5))
    fr[:, 1] = 0.6
    fr[:, 3] = 0.4
    fr[0, 2], fr[0, 3] = 0.1, 0.3  # water present in 1 of 10 images
    fr[1, 4] = 0.001  # below min_frac: not present
    rf = repeat_factors(fr, thresh=0.4, min_frac=0.005)
    assert math.isclose(rf.class_image_freq[2], 0.1)
    assert math.isclose(rf.class_factor[2], 2.0)  # sqrt(0.4 / 0.1)
    assert rf.class_factor[1] == 1.0 and rf.class_factor[4] == 1.0
    assert math.isclose(rf.image[0], 2.0) and np.all(rf.image[1:] == 1.0)
    assert set(rf.summary()) >= {"class_repeat_factor", "mean_image_factor"}
    with pytest.raises(ValueError):
        repeat_factors(fr, thresh=0.0)


def test_pass_indices_stochastic_rounding() -> None:
    f = np.array([1.0, 2.0, 1.5, 1.0])
    counts = np.zeros(4)
    n_pass = 400
    for p in range(n_pass):
        idx = pass_indices(4, np.random.default_rng(p), f)
        counts += np.bincount(idx, minlength=4)
    assert counts[0] == n_pass and counts[1] == 2 * n_pass and counts[3] == n_pass
    assert abs(counts[2] / n_pass - 1.5) < 0.1
    with pytest.raises(ValueError):
        pass_indices(4, np.random.default_rng(0), np.array([0.5, 1, 1, 1]))


def test_epoch_step_range_and_helpers() -> None:
    assert epoch_step_range(0, 10, 0, 25) == (0, 10)
    assert epoch_step_range(1, 10, 13, 25) == (13, 20)  # resumed mid-epoch
    assert epoch_step_range(2, 10, 20, 25) == (20, 25)  # capped by total
    s0, s1 = epoch_step_range(3, 10, 25, 25)
    assert s1 <= s0
    assert n_epochs_for(25, 10) == 3 and stream_length(25, 4) == 100


def test_parse_deadline() -> None:
    now = dt.datetime(2026, 9, 30, 4, 50, 0)
    assert parse_deadline(None) is None
    assert parse_deadline("09:25", now) == dt.datetime(2026, 9, 30, 9, 25).timestamp()
    assert parse_deadline("2026-09-30T09:25", now) == dt.datetime(2026, 9, 30, 9, 25).timestamp()


def test_atomic_save_replaces_file(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("old")
    assert atomic_save(lambda o, q: q.write_text(o), "new", p)
    assert p.read_text() == "new" and not (tmp_path / "x.json.tmp").exists()


def _args(root: Path, out: Path, extra: list[str]):
    from metagross.train.train_seg import build_argparser, resolve_args

    return resolve_args(
        build_argparser().parse_args(
            ["--smoke", "--data", "rugd5", "--data-root", str(root), "--init", "none", "--img", "32x48", "--bs", "2",
             "--train-subset", "0", "--val-subset", "0", "--max-iters", "0", "--out", str(out), "--aug", "robust", *extra]
        )
    )


def test_stream_training_deadline_and_exact_resume(tmp_path: Path) -> None:
    """Repeat sampling + 2-step epochs; a past deadline stops after one step; resume finishes the plan."""
    from metagross.train.synthetic import write_synthetic_dataset
    from metagross.train.train_seg import train

    root = write_synthetic_dataset(tmp_path / "data", "rugd5", n_per_split=6, hw=(48, 64))
    extra = ["--epochs", "2", "--epoch-iters", "2", "--sampling", "repeat", "--rfs-thresh", "0.5"]
    past = (dt.datetime.now() - dt.timedelta(minutes=1)).isoformat(timespec="seconds")
    # deadline already passed: the loop refuses to start an epoch, nothing is written
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="no epoch completed"):
        train(_args(root, out, [*extra, "--deadline", past]))
    # normal run to completion
    m = train(_args(root, out, extra))
    assert m["complete"] is True and m["global_iter"] == 4 and m["sampler"] == "stream"
    assert [r["global_iter"] for r in m["history"]] == [2, 4]
    assert m["repeat_factor_sampling"]["mean_image_factor"] >= 1.0
    # resume of a finished run: nothing to do, history preserved
    m2 = train(_args(root, out, [*extra, "--resume"]))
    assert len(m2["history"]) == 2 and m2["global_iter"] == 4


class _FakeClock:
    """time module stand-in: time() jumps far ahead after ``n_calls`` calls (triggers --deadline)."""

    def __init__(self, n_calls: int) -> None:
        import time as _t

        self._t, self.n, self.calls = _t, n_calls, 0
        self.perf_counter, self.sleep = _t.perf_counter, _t.sleep

    def time(self) -> float:
        self.calls += 1
        return 0.0 if self.calls <= self.n else 1e12


def test_deadline_mid_epoch_then_resume_matches_uninterrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop after 2 of 3 steps of epoch 0 (deadline), resume: final weights equal an uninterrupted run."""
    from metagross.train import train_seg
    from metagross.train.synthetic import write_synthetic_dataset

    root = write_synthetic_dataset(tmp_path / "data", "rugd5", n_per_split=6, hw=(48, 64))
    extra = ["--epochs", "2", "--epoch-iters", "3", "--sampling", "repeat", "--rfs-thresh", "0.5", "--threads", "1"]
    ref = train_seg.train(_args(root, tmp_path / "ref", extra))
    assert ref["global_iter"] == 6

    cut = tmp_path / "cut"
    # time() calls: epoch-0 start check, then one per step -> stop right after step 2
    monkeypatch.setattr(train_seg, "time", _FakeClock(n_calls=2))
    m = train_seg.train(_args(root, cut, [*extra, "--deadline", "2100-01-01T00:00"]))
    assert m["stopped_by_deadline"] is True and m["global_iter"] == 2 and m["complete"] is False
    monkeypatch.undo()
    m2 = train_seg.train(_args(root, cut, [*extra, "--resume"]))
    assert m2["global_iter"] == 6 and m2["complete"] is True
    a = torch.load(tmp_path / "ref" / "last.pt", weights_only=False)["model"]
    b = torch.load(cut / "last.pt", weights_only=False)["model"]
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-5), k
