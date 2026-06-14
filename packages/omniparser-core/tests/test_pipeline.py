"""Tests for :func:`omniparser.pipeline.merge_proposals`.

These exercise the OCR/icon merge logic without needing any models — the
merge step is pure Python operating on dict structures.
"""

from __future__ import annotations

from omniparser.pipeline import merge_proposals


def _icon(bbox: list[float]) -> dict:
    return {
        "type": "icon",
        "bbox": bbox,
        "interactivity": True,
        "content": None,
        "source": "box_yolo_content_yolo",
    }


def _ocr(bbox: list[float], text: str) -> dict:
    return {
        "type": "text",
        "bbox": bbox,
        "interactivity": False,
        "content": text,
        "source": "box_ocr_content_ocr",
    }


class TestMergeProposals:
    def test_no_proposals_returns_empty(self) -> None:
        assert merge_proposals([], [], iou_threshold=0.5) == []

    def test_ocr_only_passes_through(self) -> None:
        ocr = [_ocr([0, 0, 0.1, 0.05], "Hello")]
        out = merge_proposals([], ocr, iou_threshold=0.5)
        assert out == ocr

    def test_icon_only_passes_through(self) -> None:
        icons = [_icon([0.1, 0.1, 0.2, 0.2])]
        out = merge_proposals(icons, [], iou_threshold=0.5)
        assert len(out) == 1
        assert out[0]["type"] == "icon"
        assert out[0]["source"] == "box_yolo_content_yolo"

    def test_nms_drops_larger_duplicate_icon(self) -> None:
        # Two near-identical icons; the larger one should be dropped.
        small = _icon([0.1, 0.1, 0.2, 0.2])
        big = _icon([0.09, 0.09, 0.21, 0.21])
        out = merge_proposals([small, big], [], iou_threshold=0.5)
        assert len(out) == 1
        # Surviving box is the smaller one.
        assert out[0]["bbox"] == small["bbox"]

    def test_ocr_inside_icon_gets_absorbed_into_content(self) -> None:
        icon = _icon([0.0, 0.0, 0.2, 0.1])
        ocr = _ocr([0.05, 0.02, 0.15, 0.08], "Submit")
        out = merge_proposals([icon], [ocr], iou_threshold=0.5)
        # OCR text should no longer appear as a standalone element.
        assert all(e["type"] != "text" for e in out)
        # The icon should now carry the OCR text in its content.
        icons = [e for e in out if e["type"] == "icon"]
        assert len(icons) == 1
        assert icons[0]["content"] == "Submit"
        assert icons[0]["source"] == "box_yolo_content_ocr"

    def test_icon_inside_ocr_drops_icon(self) -> None:
        # OCR is the bigger box; icon sits fully inside it. Icon is dropped.
        ocr = _ocr([0.0, 0.0, 0.5, 0.5], "long label")
        icon = _icon([0.1, 0.1, 0.2, 0.2])
        out = merge_proposals([icon], [ocr], iou_threshold=0.5)
        assert len(out) == 1
        assert out[0]["type"] == "text"
