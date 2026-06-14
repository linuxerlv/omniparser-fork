"""Tests for omniparser_eval.scoring."""

from __future__ import annotations

from omniparser_eval.scoring import (
    Summary,
    compute_correctness,
    point_in_bbox,
    summarize,
)


class TestPointInBbox:
    def test_point_inside_bbox(self) -> None:
        assert point_in_bbox([100, 100], [50, 50, 150, 150]) is True

    def test_point_on_corner_inclusive(self) -> None:
        assert point_in_bbox([50, 50], [50, 50, 150, 150]) is True
        assert point_in_bbox([150, 150], [50, 50, 150, 150]) is True

    def test_point_outside_bbox(self) -> None:
        assert point_in_bbox([10, 10], [50, 50, 150, 150]) is False

    def test_none_inputs(self) -> None:
        assert point_in_bbox(None, [0, 0, 10, 10]) is False
        assert point_in_bbox([5, 5], None) is False
        assert point_in_bbox(None, None) is False

    def test_short_vectors(self) -> None:
        assert point_in_bbox([1], [0, 0, 10, 10]) is False
        assert point_in_bbox([1, 1], [0, 0, 10]) is False


class TestComputeCorrectness:
    def test_positive_correct(self) -> None:
        record = {"gt_type": "positive", "bbox": [50, 50, 150, 150]}
        pred = {"result": "positive", "point": [100, 100]}
        assert compute_correctness(record, pred) == "correct"  # type: ignore[arg-type]

    def test_positive_wrong_outside_bbox(self) -> None:
        record = {"gt_type": "positive", "bbox": [50, 50, 150, 150]}
        pred = {"result": "positive", "point": [10, 10]}
        assert compute_correctness(record, pred) == "wrong"  # type: ignore[arg-type]

    def test_positive_wrong_when_negative_returned(self) -> None:
        record = {"gt_type": "positive", "bbox": [50, 50, 150, 150]}
        pred = {"result": "negative", "point": None}
        assert compute_correctness(record, pred) == "wrong"  # type: ignore[arg-type]

    def test_negative_correct(self) -> None:
        record = {"gt_type": "negative", "bbox": None}
        pred = {"result": "negative", "point": None}
        assert compute_correctness(record, pred) == "negative_correct"  # type: ignore[arg-type]

    def test_negative_wrong(self) -> None:
        record = {"gt_type": "negative", "bbox": None}
        pred = {"result": "positive", "point": [50, 50]}
        assert compute_correctness(record, pred) == "negative_wrong"  # type: ignore[arg-type]

    def test_failed_when_pred_is_none(self) -> None:
        record = {"gt_type": "positive", "bbox": [0, 0, 100, 100]}
        assert compute_correctness(record, None) == "failed"  # type: ignore[arg-type]

    def test_failed_when_pred_result_is_failed(self) -> None:
        record = {"gt_type": "positive", "bbox": [0, 0, 100, 100]}
        pred = {"result": "failed", "point": None}
        assert compute_correctness(record, pred) == "failed"  # type: ignore[arg-type]


class TestSummarize:
    def test_empty(self) -> None:
        s = summarize([])
        assert s.total == 0
        assert s.scored == 0
        assert s.overall == 0.0

    def test_overall_accuracy(self) -> None:
        pairs = [
            ({"group": "A", "platform": "x", "ui_type": "icon"}, "correct"),
            ({"group": "A", "platform": "x", "ui_type": "icon"}, "wrong"),
            ({"group": "B", "platform": "y", "ui_type": "text"}, "correct"),
        ]
        s = summarize(pairs)  # type: ignore[arg-type]
        assert s.total == 3
        assert s.correct == 2
        assert s.wrong == 1
        assert s.scored == 3
        assert s.overall == 2 / 3

    def test_failed_excluded_from_scored(self) -> None:
        pairs = [
            ({"group": "A"}, "correct"),
            ({"group": "A"}, "failed"),
        ]
        s = summarize(pairs)  # type: ignore[arg-type]
        assert s.total == 2
        assert s.failed == 1
        assert s.scored == 1
        assert s.overall == 1.0

    def test_breakdown_keys(self) -> None:
        pairs = [
            ({"group": "Creative", "platform": "macos", "ui_type": "icon"}, "correct"),
            ({"group": "Creative", "platform": "macos", "ui_type": "icon"}, "wrong"),
        ]
        s = summarize(pairs)  # type: ignore[arg-type]
        assert "group=Creative" in s.by_key
        assert s.by_key["group=Creative"]["correct"] == 1
        assert s.by_key["group=Creative"]["wrong"] == 1

    def test_negative_correct_counts_toward_accuracy(self) -> None:
        pairs = [
            ({"group": "A"}, "negative_correct"),
            ({"group": "A"}, "correct"),
        ]
        s = summarize(pairs)  # type: ignore[arg-type]
        assert s.overall == 1.0


class TestSummary:
    def test_default_state(self) -> None:
        s = Summary()
        assert s.total == 0
        assert s.overall == 0.0
        assert s.by_key == {}
