"""The optional AWS teacher experiment runs end-to-end with a stub backbone (no timm needed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_teacher():
    spec = importlib.util.spec_from_file_location("teacher_dinov2", REPO_ROOT / "aws" / "teacher_dinov2.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_teacher_stub_runs(tmp_path: Path) -> None:
    from metagross.train.synthetic import write_synthetic_dataset

    t = _load_teacher()
    root = write_synthetic_dataset(tmp_path / "offroad5", "offroad5", n_per_split=6, hw=(56, 70))
    out = tmp_path / "teacher.json"
    args = t.build_argparser().parse_args(
        ["--data-root", str(root), "--backbone", "stub", "--img", "56x70", "--epochs", "1", "--bs", "2", "--workers", "0", "--device", "cpu", "--out", str(out)]
    )
    res = t.run(args)
    assert out.exists() and json.loads(out.read_text())["backbone"]["weights"] == "stub"
    assert res["best_val"]["overall"]["miou"] is not None
    assert set(res["best_val"]["per_source"]) == {"rugd", "rellis", "goose"}
    with pytest.raises(ValueError):
        t.run(t.build_argparser().parse_args(["--data-root", str(root), "--backbone", "stub", "--img", "50x70", "--out", str(out)]))
