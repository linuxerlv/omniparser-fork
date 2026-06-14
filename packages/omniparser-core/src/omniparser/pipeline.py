"""Parse-screen pipeline orchestration.

This module is the heart of omniparser-core. It composes:

* a :class:`omniparser.detection.Detector` (icon proposals)
* an :class:`omniparser.ocr.OcrBackend` (text proposals)
* a :class:`omniparser.captioning.Captioner` (icon descriptions)

into a single ``parse_screen`` function. None of the heavy ML deps are
imported until ``parse_screen`` is actually called — geometry-only
operations (e.g. :func:`merge_proposals`) work on plain Python lists.

Replaces upstream's ``get_som_labeled_img`` and the OCR-merging half of
``remove_overlap_new``. The 13-arg upstream signature is replaced with a
single :class:`OmniparserConfig` value object plus the three injected
backends.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from omniparser.geometry import box_area, int_box_area, iou, is_inside

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from omniparser.detection import Detector
    from omniparser.ocr import OcrBackend

__all__ = [
    "OmniparserConfig",
    "ParseResult",
    "ParsedElement",
    "merge_proposals",
    "parse_screen",
]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OmniparserConfig:
    """Frozen configuration for one ``parse_screen`` call.

    Replaces upstream ``get_som_labeled_img``'s 13 positional arguments.
    The renamed ``box_threshold`` field fixes upstream's ``BOX_TRESHOLD``
    typo (patch F).
    """

    box_threshold: float = 0.05
    iou_threshold: float = 0.7
    text_threshold: float = 0.8
    use_paddleocr: bool = False
    use_local_semantics: bool = True
    output_coord_in_ratio: bool = True
    scale_img: bool = False
    imgsz: tuple[int, int] | None = None
    batch_size: int = 128
    text_scale: float = 0.4
    text_padding: int = 5
    prompt: str | None = None
    draw_bbox_config: dict[str, Any] | None = None


class ParsedElement(TypedDict):
    """One element on the parsed screen, either OCR text or detected icon."""

    type: str  # "text" or "icon"
    bbox: list[float]  # normalised xyxy
    interactivity: bool
    content: str | None
    source: str  # "box_ocr_content_ocr" / "box_yolo_content_yolo" / "box_yolo_content_ocr"


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Result of a single ``parse_screen`` call.

    Attributes:
        annotated_image_b64: PNG of the input with boxes drawn, base64-encoded.
        label_coordinates: ``{label: (x, y, w, h)}`` from the annotator.
        elements: structured list of OCR + icon proposals after merging.
    """

    annotated_image_b64: str
    label_coordinates: dict[str, Any]
    elements: list[ParsedElement] = field(default_factory=list)


def merge_proposals(  # noqa: PLR0912 - branches map 1:1 to merge cases (NMS, ocr-in-icon, icon-in-ocr, fall-through)
    icon_boxes: list[ParsedElement],
    ocr_boxes: list[ParsedElement],
    *,
    iou_threshold: float,
) -> list[ParsedElement]:
    """Merge YOLO icon proposals with OCR text proposals.

    Replaces upstream ``remove_overlap_new``. The contract:

    * OCR boxes are kept verbatim.
    * Icon boxes are dropped if they're a near-duplicate of a *larger*
      icon box (NMS-style).
    * If an OCR box sits inside an icon box, its text is folded into the
      icon's ``content`` and the OCR box is removed (the icon becomes a
      labelled button rather than a separate text + icon pair).
    * If the icon box sits inside an OCR box, the icon is dropped (the
      OCR box subsumes it).

    Inputs are :class:`ParsedElement` dicts; outputs are too.
    """
    filtered: list[ParsedElement] = list(ocr_boxes)

    for i, box1_elem in enumerate(icon_boxes):
        box1 = box1_elem["bbox"]
        # NMS against other icon proposals.
        keep = True
        for j, box2_elem in enumerate(icon_boxes):
            if i == j:
                continue
            box2 = box2_elem["bbox"]
            if iou(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                keep = False
                break
        if not keep:
            continue

        # Resolve overlaps with OCR boxes.
        if not ocr_boxes:
            filtered.append(box1_elem)
            continue

        box_added = False
        ocr_labels = ""
        for box3_elem in ocr_boxes:
            if box_added:
                break
            box3 = box3_elem["bbox"]
            if is_inside(box3, box1):
                # OCR text inside icon: absorb the text into the icon's content.
                ocr_labels += (box3_elem["content"] or "") + " "
                # Remove the OCR element from `filtered` if still present.
                try:
                    filtered.remove(box3_elem)
                except ValueError:
                    # Already removed by an earlier icon; expected, not an error.
                    _logger.debug("OCR element %r already absorbed by another icon", box3_elem)
            elif is_inside(box1, box3):
                # Icon inside OCR: drop the icon. OCR boxes don't overlap
                # each other, so we can short-circuit the rest.
                box_added = True
                break

        if not box_added:
            if ocr_labels:
                filtered.append(
                    {
                        "type": "icon",
                        "bbox": list(box1_elem["bbox"]),
                        "interactivity": True,
                        "content": ocr_labels.strip() or None,
                        "source": "box_yolo_content_ocr",
                    }
                )
            else:
                filtered.append(
                    {
                        "type": "icon",
                        "bbox": list(box1_elem["bbox"]),
                        "interactivity": True,
                        "content": None,
                        "source": "box_yolo_content_yolo",
                    }
                )
    return filtered


def parse_screen(
    image: PILImage,
    *,
    detector: Detector,
    ocr_backend: OcrBackend | None,
    captioner_bundle: dict[str, Any] | None,
    config: OmniparserConfig | None = None,
) -> ParseResult:
    """Run the full OmniParser pipeline on a single image.

    Args:
        image: input screenshot as a PIL image (any mode; coerced to RGB).
        detector: any object satisfying :class:`Detector`.
        ocr_backend: any object satisfying :class:`OcrBackend`, or ``None``
            to skip OCR entirely.
        captioner_bundle: ``{'model': ..., 'processor': ...}`` dict, or
            ``None`` to skip icon captioning.
        config: :class:`OmniparserConfig`; defaults are used if omitted.

    Returns:
        :class:`ParseResult`.
    """
    import numpy as np
    import torch
    from PIL import Image as _Image
    from torchvision.ops import box_convert

    from omniparser.annotation import annotate
    from omniparser.captioning import caption_icons, caption_icons_phi3v

    cfg = config or OmniparserConfig()

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    imgsz = cfg.imgsz or (h, w)

    # 1. Detect icons.
    boxes_xyxy_pixels, _conf = detector.predict(
        image,
        box_threshold=cfg.box_threshold,
        iou_threshold=0.1,
        imgsz=imgsz if cfg.scale_img else None,
    )
    boxes_xyxy = (boxes_xyxy_pixels / np.array([w, h, w, h], dtype=np.float32)).tolist()

    # 2. Run OCR (optional).
    if ocr_backend is not None:
        ocr_texts, ocr_boxes_pixels = ocr_backend.read(image, text_threshold=cfg.text_threshold)
        ocr_boxes_normalised = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in ocr_boxes_pixels]
    else:
        ocr_texts, ocr_boxes_normalised = [], []

    image_np = np.asarray(image)

    # 3. Build structured proposals and merge.
    ocr_elements: list[ParsedElement] = [
        {
            "type": "text",
            "bbox": list(box),
            "interactivity": False,
            "content": txt,
            "source": "box_ocr_content_ocr",
        }
        for box, txt in zip(ocr_boxes_normalised, ocr_texts, strict=False)
        if int_box_area(box, w, h) > 0
    ]
    icon_elements: list[ParsedElement] = [
        {
            "type": "icon",
            "bbox": list(box),
            "interactivity": True,
            "content": None,
            "source": "box_yolo_content_yolo",
        }
        for box in boxes_xyxy
        if int_box_area(box, w, h) > 0
    ]

    merged = merge_proposals(icon_elements, ocr_elements, iou_threshold=cfg.iou_threshold)

    # Sort so that boxes with content come first; record where icons begin.
    merged.sort(key=lambda e: e["content"] is None)
    starting_idx = next((i for i, e in enumerate(merged) if e["content"] is None), -1)
    if starting_idx == -1:
        starting_idx = len(merged)
    boxes_for_caption = torch.tensor([e["bbox"] for e in merged], dtype=torch.float32)
    _logger.debug("merge_proposals → %d total, first icon at idx %d", len(merged), starting_idx)

    # 4. Caption icons (optional).
    if cfg.use_local_semantics and captioner_bundle is not None:
        caption_model = captioner_bundle["model"]
        if "phi3_v" in caption_model.config.model_type:
            captions = caption_icons_phi3v(
                boxes_for_caption.tolist(),
                ocr_boxes_normalised,
                image_np,
                captioner_bundle,
            )
        else:
            captions = caption_icons(
                boxes_for_caption.tolist(),
                starting_idx,
                image_np,
                captioner_bundle,
                prompt=cfg.prompt,
                batch_size=cfg.batch_size,
            )
        # Fill in icon captions in place.
        cap_iter = iter(captions)
        for elem in merged:
            if elem["content"] is None:
                elem["content"] = next(cap_iter, None)

    # 5. Render annotated image.
    boxes_cxcywh = box_convert(boxes=boxes_for_caption, in_fmt="xyxy", out_fmt="cxcywh")
    phrases: list[str | int] = list(range(len(boxes_cxcywh)))
    if cfg.draw_bbox_config:
        annotated_frame, label_coordinates = annotate(
            image_np, boxes_cxcywh, phrases=phrases, **cfg.draw_bbox_config
        )
    else:
        annotated_frame, label_coordinates = annotate(
            image_np,
            boxes_cxcywh,
            phrases=phrases,
            text_scale=cfg.text_scale,
            text_padding=cfg.text_padding,
        )

    pil_img = _Image.fromarray(annotated_frame)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    if cfg.output_coord_in_ratio:
        label_coordinates = {
            k: [v[0] / w, v[1] / h, v[2] / w, v[3] / h] for k, v in label_coordinates.items()
        }

    return ParseResult(
        annotated_image_b64=encoded,
        label_coordinates=label_coordinates,
        elements=merged,
    )
