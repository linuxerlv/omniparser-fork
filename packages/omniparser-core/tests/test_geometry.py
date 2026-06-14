"""Pure-function tests for :mod:`omniparser.geometry`.

These run without torch / opencv / ultralytics, so the geometry module
is the right place to enforce a high coverage floor.
"""

from __future__ import annotations

import math

import pytest

from omniparser.geometry import (
    box_area,
    int_box_area,
    intersection_area,
    iou,
    is_inside,
)


class TestBoxArea:
    def test_unit_box(self) -> None:
        assert box_area((0, 0, 1, 1)) == 1

    def test_rectangle(self) -> None:
        assert box_area((10, 20, 40, 80)) == pytest.approx(30 * 60)

    def test_zero_area_collapsed_box(self) -> None:
        assert box_area((5, 5, 5, 5)) == 0

    def test_inverted_box_returns_negative(self) -> None:
        # Inverted boxes are caller error; we don't silently fix them, but
        # we document the behaviour: signed area passes through.
        assert box_area((10, 10, 0, 0)) == 100  # (-10) * (-10)


class TestIntersection:
    def test_disjoint_boxes(self) -> None:
        assert intersection_area((0, 0, 1, 1), (10, 10, 20, 20)) == 0

    def test_identical_boxes(self) -> None:
        assert intersection_area((0, 0, 4, 4), (0, 0, 4, 4)) == 16

    def test_touching_at_edge_is_zero(self) -> None:
        assert intersection_area((0, 0, 5, 5), (5, 0, 10, 5)) == 0

    def test_partial_overlap(self) -> None:
        assert intersection_area((0, 0, 4, 4), (2, 2, 6, 6)) == 4


class TestIoU:
    def test_identical_boxes_is_one(self) -> None:
        assert iou((0, 0, 4, 4), (0, 0, 4, 4)) == pytest.approx(1.0, abs=1e-3)

    def test_disjoint_boxes_is_zero(self) -> None:
        assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == pytest.approx(0.0)

    def test_zero_area_input_returns_finite(self) -> None:
        # Degenerate: should not raise ZeroDivisionError thanks to epsilon.
        result = iou((0, 0, 0, 0), (1, 1, 2, 2))
        assert math.isfinite(result)

    def test_return_max_treats_containment_as_one(self) -> None:
        # box1 fully inside box2 → upstream-style "max" form returns 1.0
        # because ratio_of_box1 == 1.
        big = (0, 0, 10, 10)
        small = (1, 1, 2, 2)
        assert iou(small, big) == pytest.approx(1.0, abs=1e-3)

    def test_standard_iou_for_contained_box(self) -> None:
        big = (0, 0, 10, 10)
        small = (1, 1, 2, 2)
        # standard IoU = area(small) / area(big) = 1/100
        assert iou(small, big, return_max=False) == pytest.approx(0.01, abs=1e-3)


class TestIsInside:
    def test_obvious_containment(self) -> None:
        assert is_inside((1, 1, 2, 2), (0, 0, 10, 10))

    def test_disjoint(self) -> None:
        assert not is_inside((0, 0, 1, 1), (5, 5, 10, 10))

    def test_zero_area_box1(self) -> None:
        assert not is_inside((5, 5, 5, 5), (0, 0, 10, 10))

    def test_threshold_boundary(self) -> None:
        # box1 has 10 units of intersection, area 10 → ratio = 1.0,
        # comfortably above the 0.80 default.
        assert is_inside((1, 0, 6, 2), (0, 0, 10, 5))

    def test_below_threshold_is_not_inside(self) -> None:
        # box1 area = 100, intersection = 50 → ratio = 0.5 < 0.80
        assert not is_inside((0, 0, 10, 10), (5, 0, 15, 10))


class TestIntBoxArea:
    def test_normalised_box_to_pixels(self) -> None:
        # A box covering 10% by 20% of a 1000x500 image = 100 by 100 pixels.
        assert int_box_area((0.0, 0.0, 0.1, 0.2), 1000, 500) == 100 * 100

    def test_zero_area_box(self) -> None:
        assert int_box_area((0.5, 0.5, 0.5, 0.5), 1920, 1080) == 0
