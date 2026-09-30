"""Dataset indexing, per-source tagging, deterministic subsets and augmentations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metagross.train import augment as A
from metagross.train.data_index import IGNORE_INDEX, detect_layout, infer_source, list_fingerprint, list_samples, subset
from metagross.train.synthetic import synthetic_pair, write_synthetic_dataset


@pytest.fixture(scope="module")
def rugd_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_dataset(tmp_path_factory.mktemp("rugd5"), "rugd5", n_per_split=6)


@pytest.fixture(scope="module")
def offroad_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_dataset(tmp_path_factory.mktemp("offroad5"), "offroad5", n_per_split=6)


def test_layout_detection(rugd_root: Path, offroad_root: Path) -> None:
    assert detect_layout(rugd_root / "val") == "rugd5"
    assert detect_layout(offroad_root / "val") == "offroad5"
    with pytest.raises(FileNotFoundError):
        detect_layout(rugd_root / "nope")


def test_list_samples_pairs_and_sources(rugd_root: Path, offroad_root: Path) -> None:
    r = list_samples(rugd_root, "train")
    assert len(r) == 6 and all(s.source == "rugd" for s in r)
    assert all(s.mask.stem == s.image.stem + "_group5" for s in r)
    o = list_samples(offroad_root, "train")
    assert sorted({s.source for s in o}) == ["goose", "rellis", "rugd"]
    goose = [s for s in o if s.source == "goose"]
    assert goose and all(s.mask.stem.endswith("_group5b") for s in goose)
    assert goose[0].sequence == "2022-11-11_aying"
    assert [s.image for s in o] == sorted(s.image for s in o)  # deterministic order


def test_infer_source_rules() -> None:
    assert infer_source("creek_00001", "creek_00001_group5") == "rugd"
    assert infer_source("frame001604-1581623950_749", "frame001604-1581623950_749_group5") == "rellis"
    assert infer_source("2023-05-15_neubiberg_rain__0111_x_windshield_vis", "2023-05-15_neubiberg_rain__0111_x_group5b") == "goose"


def test_subset_is_seeded(rugd_root: Path) -> None:
    s = list_samples(rugd_root, "train")
    a, b, c = subset(s, 3, seed=1), subset(s, 3, seed=1), subset(s, 3, seed=2)
    assert a == b and len(a) == 3
    assert list_fingerprint(a) == list_fingerprint(b)
    assert subset(s, 0) == s and subset(s, 100) == s
    assert a != c or list_fingerprint(a) == list_fingerprint(c)  # different seeds may coincide on tiny sets


@pytest.mark.parametrize("policy", ["clean", "robust", "none"])
def test_augment_shapes_and_labels(policy: str) -> None:
    rng = np.random.default_rng(0)
    img, lab = synthetic_pair((60, 90), rng)
    cfg = A.AugConfig(out_hw=(48, 64), policy=policy)
    for k in range(10):
        o_img, o_lab = A.augment(img, lab, cfg, np.random.default_rng(k))
        assert o_img.shape == (48, 64, 3) and o_img.dtype == np.uint8
        assert o_lab.shape == (48, 64) and o_lab.dtype == np.uint8
        assert set(np.unique(o_lab)) <= {0, 1, 2, 3, 4, IGNORE_INDEX}


def test_augment_deterministic_given_seed() -> None:
    img, lab = synthetic_pair((60, 90), np.random.default_rng(3))
    cfg = A.AugConfig(out_hw=(48, 64), policy="robust")
    a = A.augment(img, lab, cfg, np.random.default_rng([0, 1, 2]))
    b = A.augment(img, lab, cfg, np.random.default_rng([0, 1, 2]))
    c = A.augment(img, lab, cfg, np.random.default_rng([0, 2, 2]))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_robust_ops_preserve_shape() -> None:
    rng = np.random.default_rng(0)
    img, _ = synthetic_pair((40, 56), rng)
    for fn in (A.exposure_clip, A.motion_blur, A.synthetic_flare, A.haze, A.jpeg, A.sensor_noise, A.color_jitter):
        out = fn(img, rng)
        assert out.shape == img.shape and out.dtype == np.uint8, fn.__name__
    assert A.apply_gamma(img, 2.0).mean() < img.mean() < A.apply_gamma(img, 0.5).mean()


def test_scale_crop_pads_with_ignore() -> None:
    img = np.full((10, 10, 3), 100, np.uint8)
    lab = np.full((10, 10), 3, np.uint8)
    # Out size much larger than the scaled input -> padding must be IGNORE_INDEX.
    o_img, o_lab = A.random_scale_crop(img, lab, (40, 200), np.random.default_rng(0))
    assert o_lab.shape == (40, 200)
    assert (o_lab == IGNORE_INDEX).any()
    assert set(np.unique(o_lab)) <= {3, IGNORE_INDEX}


def test_torch_dataset_and_class_weights(rugd_root: Path) -> None:
    torch = pytest.importorskip("torch")
    from metagross.train.datasets import SegDataset, class_frequencies, enet_class_weights

    samples = list_samples(rugd_root, "train")
    ds = SegDataset(samples, (32, 40), "robust", seed=5)
    x, y, i = ds[2]
    assert x.shape == (3, 32, 40) and x.dtype == torch.float32
    assert y.shape == (32, 40) and y.dtype == torch.int64 and i == 2
    x2, _, _ = ds[2]
    assert torch.equal(x, x2)
    ds.set_epoch(1)
    assert not torch.equal(x, ds[2][0])
    freq = class_frequencies(samples)
    assert freq.shape == (5,) and abs(freq.sum() - 1) < 1e-9
    w = enet_class_weights(freq)
    assert abs(w.mean() - 1) < 1e-5
    assert w[np.argmin(freq)] == w.max()  # rarest class gets the largest weight
