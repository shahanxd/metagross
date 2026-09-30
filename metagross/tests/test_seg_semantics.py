"""Segmenter (ONNX Runtime) shapes, ADE20K->5 mapping table, config resolution, import hygiene."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from metagross.autonomy.perception import semantics as S

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _conv_model(path: Path, n_out: int, hw: tuple[int, int] | None, pool: int = 1, bias_class: int | None = None, seed: int = 0) -> Path:
    """Tiny ONNX graph: [AveragePool(pool)] -> 1x1 Conv(3 -> n_out). Static or dynamic HxW."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n_out, 3, 1, 1)) * 0.1).astype(np.float32)
    b = np.zeros(n_out, np.float32)
    if bias_class is not None:
        b[bias_class] = 20.0  # dominate every pixel
    h, wd = hw if hw else ("H", "W")
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, h, wd])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_out, None, None])
    nodes, conv_in = [], "input"
    if pool > 1:
        nodes.append(helper.make_node("AveragePool", ["input"], ["pooled"], kernel_shape=[pool, pool], strides=[pool, pool]))
        conv_in = "pooled"
    nodes.append(helper.make_node("Conv", [conv_in, "W", "B"], ["logits"]))
    graph = helper.make_graph(nodes, "tiny", [x], [y], initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _rgb(h: int = 60, w: int = 90, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- mapping table
def test_ade_mapping_verified_ids() -> None:
    lut = S.ade20k_lut()
    assert lut.shape == (150,) and lut.dtype == np.uint8
    stable = [6, 11, 52, 54, 3]
    unstable = [9, 13, 29, 46, 91, 94]
    water = [21, 26, 60, 109, 113, 128]
    background = [2, 16, 68]
    obstacle = [0, 1, 4, 12, 17, 20, 32, 34, 43, 69, 72, 80, 83, 93, 102, 126, 127]
    for ids, cls in ((stable, 4), (unstable, 3), (water, 2), (background, 0), (obstacle, 1)):
        assert all(lut[i] == cls for i in ids), cls
    listed = set(S.ADE20K_TO_SEM5)
    unlisted = [i for i in range(150) if i not in listed]
    assert unlisted and all(lut[i] == S.SEM_OBSTACLE for i in unlisted)  # conservative default
    assert len(set(stable + unstable + water + background + obstacle)) == len(listed)


def test_group_matrix_partitions_probability() -> None:
    m = S.ade20k_group_matrix()
    assert m.shape == (5, 150)
    np.testing.assert_array_equal(m.sum(axis=0), np.ones(150))
    p = np.random.default_rng(0).dirichlet(np.ones(150), size=7).T  # (150, 7)
    np.testing.assert_allclose((m @ p).sum(axis=0), 1.0, rtol=1e-6)


def test_palette_and_colorize() -> None:
    assert S.SEM_PALETTE.shape == (5, 3)
    ids = np.array([[0, 1, 2], [3, 4, 255]], np.uint8)
    rgb = S.colorize(ids)
    assert rgb.shape == (2, 3, 3) and tuple(rgb[1, 2]) == (0, 0, 0)
    assert tuple(rgb[0, 1]) == S.SEM_COLORS[1]


def test_entropy_bounds() -> None:
    uniform = np.full((5, 2, 2), 0.2, np.float32)
    certain = np.zeros((5, 2, 2), np.float32)
    certain[3] = 1.0
    np.testing.assert_allclose(S.normalized_entropy(uniform), 1.0, atol=1e-5)
    np.testing.assert_allclose(S.normalized_entropy(certain), 0.0, atol=1e-5)


# --------------------------------------------------------------------------- segmenter
def test_direct5_shapes_and_dtypes(tmp_path: Path) -> None:
    mp = _conv_model(tmp_path / "d5.onnx", 5, (32, 48))
    seg = S.Segmenter(mp, intra_op_threads=1)
    assert seg.spec.kind == "direct5" and seg.spec.input_hw == (32, 48)
    for h, w in ((60, 90), (400, 640), (32, 48)):
        ids, ent = seg(_rgb(h, w))
        assert ids.shape == (h, w) and ids.dtype == np.uint8 and ids.max() < 5
        assert ent.shape == (h, w) and ent.dtype == np.float32
        assert 0.0 <= float(ent.min()) and float(ent.max()) <= 1.0
    assert set(seg.last_timings_ms) == {"seg_pre", "seg_infer", "seg_post", "seg_total"}
    p = seg.predict_proba(_rgb())
    assert p.shape == (5, 32, 48)
    np.testing.assert_allclose(p.sum(axis=0), 1.0, rtol=1e-5)


def test_ade150_quarter_res_and_mapping(tmp_path: Path) -> None:
    # dynamic input, logits at 1/4 resolution like SegFormer; bias forces ADE 6 (road) -> stable
    road = _conv_model(tmp_path / "ade_road.onnx", 150, None, pool=4, bias_class=6)
    (tmp_path / "ade_road.json").write_text(json.dumps({"kind": "ade150", "input_hw": [64, 96]}))
    seg = S.Segmenter(road, intra_op_threads=1)
    assert seg.spec.kind == "ade150" and seg.spec.input_hw == (64, 96)
    ids, ent = seg(_rgb(100, 150))
    assert ids.shape == (100, 150) and np.all(ids == S.SEM_STABLE)
    assert float(ent.max()) < 0.05  # one class dominates
    # an unlisted ADE class (5 = ceiling) must map to obstacle
    ceil = _conv_model(tmp_path / "ade_ceiling.onnx", 150, (64, 96), pool=4, bias_class=5)
    ids2, _ = S.Segmenter(ceil, intra_op_threads=1)(_rgb(50, 70))
    assert np.all(ids2 == S.SEM_OBSTACLE)


def test_input_override_rules(tmp_path: Path) -> None:
    dyn = _conv_model(tmp_path / "dyn.onnx", 5, None)
    seg = S.Segmenter(dyn, intra_op_threads=1, input_hw=(40, 56))
    assert seg.spec.input_hw == (40, 56)
    static = _conv_model(tmp_path / "static.onnx", 5, (32, 48))
    with pytest.raises(ValueError):
        S.Segmenter(static, intra_op_threads=1, input_hw=(40, 56))
    with pytest.raises(ValueError):
        seg(np.zeros((10, 10), np.uint8))


def test_from_config_and_missing(tmp_path: Path) -> None:
    mp = _conv_model(tmp_path / "cfg.onnx", 5, (32, 48))
    seg = S.Segmenter.from_config({"seg_model_path": str(mp), "seg_threads": 1})
    assert seg.threads == 1 and seg.model_path == mp
    with pytest.raises(FileNotFoundError):
        S.Segmenter.from_config({"seg_model_path": str(tmp_path / "missing.onnx")})
    assert S.resolve_model_path({"seg_model_path": str(mp)}) == mp


def test_constructible_by_node_wiring(tmp_path: Path) -> None:
    """AutonomyStack builds components by keyword (calib/vehicle/config/...): no required args."""
    import inspect

    required = [n for n, p in inspect.signature(S.Segmenter).parameters.items() if p.default is inspect.Parameter.empty]
    assert required == []
    mp = _conv_model(tmp_path / "node.onnx", 5, (32, 48))
    cfg = {"seg_model_path": str(mp), "seg_threads": 1, "use_semantics": True}
    seg = S.Segmenter(config=cfg)
    assert seg.threads == 1 and seg(_rgb())[0].shape == (60, 90)
    node = pytest.importorskip("metagross.autonomy.node")
    construct = getattr(node, "_construct", None)
    if construct is None:
        pytest.skip("node._construct not available")
    seg2 = construct(S.Segmenter, calib=None, vehicle=None, config=cfg, mission=None, seed=0)
    assert seg2.model_path == mp


def test_seg_eval_cli_and_model_card(tmp_path: Path) -> None:
    """seg_eval CLI end-to-end on a synthetic dataset, plus --latency-only / --figures-only and the card."""
    from metagross.eval import seg_eval
    from metagross.train import model_card
    from metagross.train.synthetic import write_synthetic_dataset

    root = write_synthetic_dataset(tmp_path / "rugd5", "rugd5", n_per_split=4, hw=(40, 56))
    mp = _conv_model(tmp_path / "m.onnx", 5, (32, 48))
    out = tmp_path / "results" / "seg_test.json"
    figs = tmp_path / "figs" / "seg_test"
    base = ["--model", str(mp), "--data-root", str(root), "--out", str(out), "--fig-prefix", str(figs), "--threads", "1"]
    assert seg_eval.main(base + ["--splits", "val", "test", "--latency-threads", "1", "--n-qualitative", "2", "--label", "unit"]) == 0
    d = json.loads(out.read_text())
    assert d["schema"] == seg_eval.SCHEMA and set(d["splits"]) == {"val", "test"}
    assert d["splits"]["val"]["n_images"] == 4 and "brightness" in d["splits"]["val"]
    assert d["latency_ms"]["camera_frame"]["image_hw"] == [400, 640]
    assert any(c["id"].endswith("val_false_safe_rate") for c in d["claims"])
    for sp in ("val", "test"):
        assert Path(f"{figs}_confusion_{sp}.png").exists() and Path(f"{figs}_grid_{sp}.png").exists()
    stamp = d["latency_ms"]["measured_utc"]
    assert seg_eval.main(base + ["--splits", "val", "--latency-only", "--latency-threads", "1"]) == 0
    assert json.loads(out.read_text())["splits"] == d["splits"]  # accuracy untouched
    assert json.loads(out.read_text())["latency_ms"]["measured_utc"] >= stamp
    Path(f"{figs}_grid_val.png").unlink()
    assert seg_eval.main(base + ["--figures-only", "--n-qualitative", "2"]) == 0
    assert Path(f"{figs}_grid_val.png").exists()
    card = tmp_path / "CARD.md"
    card.write_text(f"# x\n{model_card.BEGIN}\nold\n{model_card.END}\ntail\n", encoding="utf-8")
    model_card.update_card(card, out.parent)
    txt = card.read_text(encoding="utf-8")
    assert "old" not in txt and "| unit | rugd5 val | 4 |" in txt and txt.endswith("tail\n")


def test_semantics_never_imports_sim_eval_train() -> None:
    src = Path(S.__file__).read_text(encoding="utf-8")
    banned = ("metagross.sim", "metagross.eval", "metagross.train")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(banned), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith(banned) for a in node.names)
