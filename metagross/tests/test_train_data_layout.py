"""data_index.resolve_split_dir: tolerate one wrapper folder from a dataset zip."""

from __future__ import annotations

import shutil
from pathlib import Path

from metagross.train.data_index import list_samples, resolve_split_dir
from metagross.train.synthetic import write_synthetic_dataset


def test_direct_layout_unchanged(tmp_path: Path) -> None:
    root = write_synthetic_dataset(tmp_path / "offroad5", "offroad5", n_per_split=3, hw=(16, 20))
    assert resolve_split_dir(root, "val") == root / "val"
    assert len(list_samples(root, "val")) == 3


def test_wrapper_folder_above_splits(tmp_path: Path) -> None:
    inner = write_synthetic_dataset(tmp_path / "offroad5" / "OFFROAD5", "offroad5", n_per_split=3, hw=(16, 20))
    (tmp_path / "offroad5" / "_zips").mkdir()  # ignored: private folders do not count as wrappers
    root = tmp_path / "offroad5"
    assert resolve_split_dir(root, "train") == inner / "train"
    assert len(list_samples(root, "train")) == 3


def test_wrapper_folder_below_split(tmp_path: Path) -> None:
    src = write_synthetic_dataset(tmp_path / "src", "rugd5", n_per_split=2, hw=(16, 20))
    root = tmp_path / "rugd5"
    shutil.copytree(src / "test", root / "test" / "test")
    assert resolve_split_dir(root, "test") == root / "test" / "test"
    assert len(list_samples(root, "test")) == 2
