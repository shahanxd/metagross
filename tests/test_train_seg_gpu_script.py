"""scripts/train_seg_gpu.ps1 passes only flags that the Python entry points accept.

PowerShell cannot run in CI, so the script's command lines are checked statically: every ``--flag`` it
hands to ``train_seg``, ``export_onnx`` and ``seg_eval`` must exist, and the training command line
(with the script's defaults substituted) must parse and resolve to the intended recipe.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from metagross.train import train_seg  # noqa: E402  (after the torch skip)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (REPO / "scripts" / "train_seg_gpu.ps1").read_text(encoding="utf-8")


def _block(start: str, end: str) -> str:
    i = SCRIPT.index(start)
    return SCRIPT[i:SCRIPT.index(end, i)]


def _flags(text: str) -> set[str]:
    return set(re.findall(r'"(--[a-z][a-z0-9-]*)"', text)) | set(re.findall(r"\s(--[a-z][a-z0-9-]*)\s", text))


def _module_flags(rel: str) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[a-z][a-z0-9-]*)"', (REPO / rel).read_text(encoding="utf-8")))


def _default(name: str) -> str:
    m = re.search(rf"\[(?:int|string)\]\${name}\s*=\s*\"?([^\",)\s]+)", SCRIPT)
    assert m, name
    return m.group(1)


def test_train_flags_exist_and_parse() -> None:
    block = _block("function Get-TrainArgs", "return $a")
    flags = _flags(block)
    known = {a for act in train_seg.build_argparser()._actions for a in act.option_strings}
    assert flags and flags <= known, flags - known
    # the launched command line with the script defaults (epoch-iters, full data, full validation set)
    argv = ["--model", "lraspp", "--data", "offroad5", "--data-root", "data/offroad5", "--aug", "robust",
            "--init", "imagenet", "--epochs", _default("Epochs"), "--epoch-iters", _default("EpochIters"),
            "--bs", _default("Bs"), "--lr", "6e-4", "--img", _default("Img"), "--amp", "--device", "cuda",
            "--workers", _default("Workers"), "--threads", "2", "--train-subset", "0", "--val-subset", "0",
            "--resume", "--out", "runs/seg/lraspp_offroad5_robust"]
    assert set(a for a in argv if a.startswith("--")) == flags - {"--deadline"}
    args = train_seg.resolve_args(train_seg.build_argparser().parse_args(argv))
    assert train_seg.parse_hw(args.img) == (320, 416)
    assert args.amp and args.resume and args.epoch_iters > 0   # step-indexed sampler: exact mid-epoch resume
    assert args.bs * args.epoch_iters * args.epochs >= 20 * 8081   # at least ~20 passes over OFFROAD5 train


def test_export_and_eval_flags_exist() -> None:
    finish = SCRIPT[SCRIPT.index("# ------------------------------------------------------------------ finish"):]
    export = finish[finish.index("metagross.train.export_onnx"):finish.index('Check "export"')]
    evalc = finish[finish.index("metagross.eval.seg_eval"):finish.index('Check "seg_eval')]
    assert _flags(export) <= _module_flags("metagross/train/export_onnx.py")
    assert _flags(evalc) <= _module_flags("metagross/eval/seg_eval.py")


def test_output_names_match_onboard_segmenter_search_order() -> None:
    # the onboard Segmenter looks for models/lraspp_offroad5_robust.onnx first
    sem = (REPO / "metagross/autonomy/perception/semantics.py").read_text(encoding="utf-8")
    assert '$Name = "lraspp_offroad5_$Aug"' in SCRIPT and '$model = "models/$Name.onnx"' in SCRIPT
    assert '"lraspp_offroad5_robust.onnx"' in sem
