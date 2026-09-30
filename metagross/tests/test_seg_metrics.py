"""Metric correctness on hand-made confusion cases (metagross.eval.seg_eval)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.eval.seg_eval import SegAccumulator, confusion_matrix, false_safe_rate, image_luminance, iou_per_class, summarize


def test_confusion_counts_and_ignore() -> None:
    gt = np.array([[0, 1, 1, 255], [2, 3, 4, 4]], np.uint8)
    pr = np.array([[0, 1, 3, 4], [3, 3, 4, 1]], np.uint8)
    cm = confusion_matrix(gt, pr)
    assert cm.sum() == 7  # the 255 pixel is ignored
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[1, 3] == 1
    assert cm[2, 3] == 1 and cm[3, 3] == 1 and cm[4, 4] == 1 and cm[4, 1] == 1
    with pytest.raises(ValueError):
        confusion_matrix(gt, np.full_like(gt, 7))


def test_iou_hand_computed() -> None:
    # class 1: TP 1, FN 1 (->3), FP 1 (from 4) => IoU 1/3
    # class 3: TP 1, FP 2 (from 1 and 2)      => IoU 1/3
    # class 4: TP 1, FN 1 (->1)                => IoU 1/2 ; class 0 perfect ; class 2: TP 0 => 0
    gt = np.array([0, 1, 1, 2, 3, 4, 4], np.uint8)
    pr = np.array([0, 1, 3, 3, 3, 4, 1], np.uint8)
    iou = iou_per_class(confusion_matrix(gt, pr))
    np.testing.assert_allclose(iou, [1.0, 1 / 3, 0.0, 1 / 3, 1 / 2])
    s = summarize(confusion_matrix(gt, pr))
    assert math.isclose(s["miou"], (1 + 1 / 3 + 0 + 1 / 3 + 1 / 2) / 5)
    assert math.isclose(s["pixel_acc"], 4 / 7)
    # hazards: GT 1 (x2) and 2 (x1); predicted traversable: one obstacle px -> 3, water px -> 3
    assert math.isclose(s["false_safe_rate"], 2 / 3)
    assert math.isclose(s["false_safe_obstacle"], 1 / 2)
    assert math.isclose(s["false_safe_water"], 1.0)
    # traversable GT (3, 4, 4): one stable px predicted obstacle
    assert math.isclose(s["false_hazard_rate"], 1 / 3)


def test_absent_classes_are_none_not_zero() -> None:
    gt = np.array([3, 3, 4], np.uint8)
    pr = np.array([3, 3, 4], np.uint8)
    s = summarize(confusion_matrix(gt, pr))
    assert s["per_class_iou"]["obstacle"] is None
    assert s["miou"] == 1.0 and s["false_safe_rate"] is None
    assert s["false_safe_water"] is None


def test_false_safe_rate_only_counts_traversable_predictions() -> None:
    cm = np.zeros((5, 5), np.int64)
    cm[1, 0] = 10  # obstacle called sky: wrong, but not "false safe"
    cm[1, 2] = 10  # obstacle called water: still a hazard
    cm[1, 4] = 5  # obstacle called stable: false safe
    assert math.isclose(false_safe_rate(cm), 5 / 25)


def test_accumulator_strata() -> None:
    acc = SegAccumulator()
    gt = np.array([[1, 3], [3, 4]], np.uint8)
    good, bad = gt.copy(), np.full_like(gt, 3)
    for luma, src, pred in [(20, "rugd", bad), (25, "rugd", bad), (120, "goose", good), (130, "goose", good), (200, "rellis", good), (220, "rellis", good)]:
        acc.add(gt, pred, src, luma, entropy=np.where(pred == gt, 0.1, 0.9).astype(np.float32), sequence=f"seq_{src}")
    rep = acc.report()
    seq = rep["per_sequence"]
    assert set(seq) == {"seq_rugd", "seq_goose", "seq_rellis"}
    assert seq["seq_rugd"]["hazard_px"] == 2 and seq["seq_rugd"]["false_safe_px"] == 2
    assert seq["seq_goose"]["false_safe_px"] == 0 and seq["seq_goose"]["n_images"] == 2
    assert rep["n_images"] == 6
    assert set(rep["per_source"]) == {"rugd", "goose", "rellis"}
    assert rep["per_source"]["goose"]["miou"] == 1.0
    assert rep["brightness"]["dark"]["n_images"] == 2
    assert rep["brightness"]["dark"]["false_safe_rate"] == 1.0  # dark images were the bad ones
    assert rep["brightness"]["bright"]["false_safe_rate"] == 0.0
    assert rep["entropy"]["mean_on_wrong"] > rep["entropy"]["mean_on_correct"]
    np.testing.assert_array_equal(acc.total(), sum(acc.cms))


def test_image_luminance() -> None:
    assert image_luminance(np.zeros((4, 4, 3), np.uint8)) == 0.0
    assert abs(image_luminance(np.full((4, 4, 3), 255, np.uint8)) - 255.0) < 1e-3
