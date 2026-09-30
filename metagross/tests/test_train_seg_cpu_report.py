"""seg_cpu_report: provenance, baseline comparison on identical images, claims rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from metagross.eval.seg_eval import summarize  # noqa: E402
from metagross.train.seg_cpu_report import checkpoint_info, describe, main, update_result  # noqa: E402


def _result(cm: np.ndarray, sha: str, label: str, path: str) -> dict:
    rep = {"n_images": 10, "overall": summarize(cm), "file_list_sha": sha, "max_images": None}
    return {"schema": "metagross.seg_eval/1", "label": label, "model": {"path": path},
            "splits": {"val": rep, "test": dict(rep)},
            "claims": [{"id": "seg_cpu_val_miou", "value": 0.5, "label": "Tested", "source": "x", "note": label}]}


def _cm(diag: float, fs: int) -> np.ndarray:
    cm = np.full((5, 5), 1, np.int64) + np.eye(5, dtype=np.int64) * int(diag)
    cm[2, 4] += fs  # water predicted stable = false-safe
    return cm


def _ckpt(tmp: Path, at: int, planned: int) -> Path:
    p = tmp / "best.pt"
    meta = {"total_iters": planned, "iters_per_epoch": 250, "n_train": 4779, "n_val": 300, "img_hw": [256, 320],
            "args": {"bs": 8, "aug": "robust", "sampling": "repeat", "lr": 6e-4},
            "repeat_factor_sampling": {"mean_image_factor": 1.09}}
    torch.save({"model": {}, "meta": meta, "epoch": at // 250 - 1, "global_iter": at, "val": {"miou": 0.6, "false_safe_rate": 0.2}}, p)
    return p


def test_hazard_to_stable_is_strict_subset_of_false_safe() -> None:
    from metagross.train.seg_cpu_report import hazard_to_stable

    cm = np.zeros((5, 5), np.int64)
    cm[1] = [0, 6, 0, 3, 1]  # obstacle: 3 -> unstable, 1 -> stable
    cm[2] = [0, 0, 8, 0, 2]  # water: 2 -> stable
    r = hazard_to_stable(cm)
    assert r["obstacle_to_stable_rate"] == 0.1 and r["water_to_stable_rate"] == 0.2
    assert r["hazard_to_stable_rate"] == 3 / 20
    assert summarize(cm)["false_safe_rate"] == 6 / 20  # traversable = unstable + stable
    assert hazard_to_stable(np.eye(5, dtype=np.int64) * 0)["hazard_to_stable_rate"] is None


def test_checkpoint_info_partial_flag() -> None:
    info = checkpoint_info({"meta": {"total_iters": 4000, "args": {}}, "global_iter": 250}, None)
    assert info["partial"] is True and "PARTIAL" in describe(info)
    info = checkpoint_info({"meta": {"total_iters": 4000, "args": {}}, "global_iter": 4000}, {"history": [{"train_s_per_iter": 3.0}], "complete": True})
    assert info["partial"] is False and info["run_progress"]["last_train_s_per_iter"] == 3.0


def test_update_result_compares_only_identical_images(tmp_path: Path) -> None:
    ours = tmp_path / "seg_cpu.json"
    ours.write_text(json.dumps(_result(_cm(100, 0), "abc", "ours", "models/lraspp_offroad5_cpu.onnx")))
    smoke = tmp_path / "seg_smoke.json"
    smoke.write_text(json.dumps(_result(_cm(50, 30), "abc", "smoke", "models/lraspp_smoke.onnx")))
    other = tmp_path / "seg_zeroshot.json"
    other.write_text(json.dumps(_result(_cm(50, 30), "zzz", "zeroshot", "models/x.onnx")))
    run = tmp_path / "run"
    run.mkdir()
    out = update_result(ours, _ckpt(tmp_path, 500, 4000), run, [smoke, other, tmp_path / "missing.json"])
    assert out["training"]["partial"] is True and out["training"]["iterations_at_best"] == 500
    val = out["comparison"]["val"]
    assert val["baselines"]["smoke"]["same_images"] is True
    assert val["baselines"]["zeroshot"]["same_images"] is False and "delta_ours_minus_baseline" not in val["baselines"]["zeroshot"]
    d = val["baselines"]["smoke"]["delta_ours_minus_baseline"]
    assert d["miou"] > 0 and d["false_safe_water"] < 0 and d["water_to_stable_rate"] < 0
    ids = {c["id"]: c for c in out["claims"]}
    assert {"seg_cpu_val_iou_water", "seg_cpu_test_false_safe_water", "seg_cpu_val_miou_delta_vs_smoke", "seg_cpu_train_iters_at_best",
            "seg_cpu_test_hazard_to_stable_rate", "seg_cpu_val_hazard_to_stable_rate_delta_vs_smoke"} <= set(ids)
    assert "seg_cpu_val_miou_delta_vs_zeroshot" not in ids
    assert all(c["label"] == "Tested" for c in out["claims"])
    assert "PARTIAL" in ids["seg_cpu_val_miou"]["note"]
    # idempotent: a second pass neither duplicates rows nor stacks notes
    out2 = update_result(ours, _ckpt(tmp_path, 500, 4000), run, [smoke])
    assert len([c for c in out2["claims"] if c["id"] == "seg_cpu_val_iou_water"]) == 1
    assert out2["claims"][0]["note"].count("PARTIAL") == 1


def test_label_only_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--label-only", "--ckpt", str(_ckpt(tmp_path, 4000, 4000)), "--run-dir", str(tmp_path)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("LR-ASPP") and "4000/4000" in line and "PARTIAL" not in line
