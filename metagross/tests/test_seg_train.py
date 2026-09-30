"""Loss, schedule and model sanity + a tiny end-to-end train -> export -> Segmenter run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from metagross.train.losses import CEDiceLoss, soft_dice_loss  # noqa: E402
from metagross.train.train_seg import lr_lambda_factory, parse_hw, torch_confusion  # noqa: E402


def test_dice_perfect_and_ignore() -> None:
    target = torch.tensor([[[0, 1], [3, 255]]])
    logits = torch.full((1, 5, 2, 2), -30.0)
    for c, (i, j) in ((0, (0, 0)), (1, (0, 1)), (3, (1, 0))):
        logits[0, c, i, j] = 30.0
    logits[0, 2, 1, 1] = 30.0  # prediction on an ignored pixel must not matter
    assert float(soft_dice_loss(logits, target)) < 1e-3
    loss, parts = CEDiceLoss(torch.ones(5))(logits, target)
    assert float(loss) < 1e-3 and set(parts) == {"ce", "dice"}


def test_loss_decreases_with_better_logits() -> None:
    g = torch.Generator().manual_seed(0)
    target = torch.randint(0, 5, (2, 8, 8), generator=g)
    bad = torch.randn(2, 5, 8, 8, generator=g)
    good = bad + 5.0 * torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2)
    fn = CEDiceLoss(torch.tensor([1.0, 1.0, 3.0, 1.0, 1.0]))
    assert float(fn(good, target)[0]) < float(fn(bad, target)[0])


@pytest.mark.parametrize("sched", ["poly", "cosine"])
def test_lr_schedule(sched: str) -> None:
    f = lr_lambda_factory(total_iters=100, warmup=10, sched=sched)
    vals = [f(i) for i in range(100)]
    assert math.isclose(vals[0], 0.1) and math.isclose(vals[9], 1.0)
    assert all(a >= b - 1e-12 for a, b in zip(vals[10:], vals[11:]))  # monotone decay
    assert vals[-1] >= 0.01 - 1e-9


def test_parse_hw_and_torch_confusion() -> None:
    assert parse_hw("320x416") == (320, 416)
    gt = torch.tensor([0, 1, 255, 4])
    pr = torch.tensor([0, 3, 2, 4])
    cm = torch_confusion(gt, pr)
    assert int(cm.sum()) == 3 and int(cm[1, 3]) == 1 and int(cm[4, 4]) == 1


def test_lraspp_forward_shape() -> None:
    from metagross.train.models import build_model

    torch.manual_seed(0)
    net = build_model("lraspp", 5, init="none").eval()
    with torch.no_grad():
        y = net(torch.zeros(1, 3, 64, 80))
    assert tuple(y.shape) == (1, 5, 64, 80)


def test_end_to_end_train_export_segment(tmp_path: Path) -> None:
    """2 CPU iterations on a synthetic dataset, ONNX export, then the onboard Segmenter."""
    pytest.importorskip("onnxruntime")
    from metagross.autonomy.perception.semantics import Segmenter
    from metagross.train.export_onnx import export
    from metagross.train.synthetic import write_synthetic_dataset
    from metagross.train.train_seg import build_argparser, resolve_args, train

    root = write_synthetic_dataset(tmp_path / "data", "offroad5", n_per_split=6, hw=(48, 64))
    out = tmp_path / "run"
    args = resolve_args(
        build_argparser().parse_args(
            ["--smoke", "--data", "offroad5", "--data-root", str(root), "--init", "none", "--img", "32x48", "--bs", "2",
             "--max-iters", "2", "--train-subset", "0", "--val-subset", "0", "--out", str(out), "--aug", "robust"]
        )
    )
    m = train(args)
    assert m["smoke"] is True and m["history"][-1]["global_iter"] == 2
    assert set(m["history"][-1]["val_per_source_miou"]) == {"rugd", "rellis", "goose"}
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    # resume: nothing left to do, history preserved
    args.resume = True
    m2 = train(args)
    assert len(m2["history"]) == len(m["history"])

    onnx_path = tmp_path / "models" / "tiny_smoke.onnx"
    side = export(out / "best.pt", onnx_path, data_root=root)
    assert side["kind"] == "direct5" and side["input_hw"] == [32, 48]
    assert side["export"]["max_abs_logit_diff_vs_torch"] < 1e-3
    assert json.loads(onnx_path.with_suffix(".json").read_text())["smoke"] is True
    ids, ent = Segmenter(onnx_path, intra_op_threads=1)(np.zeros((50, 70, 3), np.uint8))
    assert ids.shape == (50, 70) and ent.shape == (50, 70)
