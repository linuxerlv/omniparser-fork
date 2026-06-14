"""Gradio demo for ``omniparser-core``.

Differences from upstream ``gradio_demo.py``:

* No model loading at import time — weights are loaded lazily on the first
  click via :func:`_get_parser` and cached for the rest of the process.
* ``share=False`` by default — opt-in via ``--share`` because the upstream
  default exposed a public ``gradio.live`` tunnel that bypasses the loopback
  binding.
* Device is resolved through :func:`omniparser.resolve_device` (CUDA → MPS →
  CPU) rather than hardcoded to ``cuda``.
* All ``print`` calls replaced with structlog (patch C).
* :func:`build_demo` is the public factory so tests can import it without
  actually launching the server.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gradio as gr
    from PIL.Image import Image as PILImage

    from omniparser import Omniparser

_logger = logging.getLogger(__name__)

_MARKDOWN = """\
# OmniParser — single-page demo

Upload a screenshot to extract OCR text and icon captions.
The first parse loads the YOLO + caption models (a few seconds);
subsequent parses are fast.
"""


@dataclass(slots=True, frozen=True)
class DemoConfig:
    """Settings for the demo app.

    All paths are required up front because the demo cannot do anything
    useful without weights.
    """

    detector_weights: Path
    captioner_weights: Path
    captioner_backend: str = "florence2"
    server_name: str = "127.0.0.1"
    server_port: int = 7861
    share: bool = False


class _ParserSlot:
    """Lazy, thread-safe holder for an :class:`Omniparser` instance.

    Gradio handlers may execute concurrently on different threads, so we
    guard the load with an :class:`threading.Lock`. Construction itself is
    expensive (loads YOLO + Florence-2), but only the first call pays for
    it.
    """

    __slots__ = ("_lock", "_parser")

    def __init__(self) -> None:
        self._parser: Omniparser | None = None
        self._lock = threading.Lock()

    def get(self, cfg: DemoConfig) -> Omniparser:
        if self._parser is not None:
            return self._parser
        with self._lock:
            if self._parser is None:
                self._parser = _build_parser(cfg)
        return self._parser


def _build_parser(cfg: DemoConfig) -> Omniparser:
    from omniparser import Omniparser, OmniparserConfig, resolve_device
    from omniparser.captioning import load_caption_model
    from omniparser.detection import UltralyticsYoloDetector
    from omniparser.ocr import EasyOcrBackend

    device = resolve_device(None)
    _logger.info(
        "demo_loading_models detector=%s captioner=%s backend=%s device=%s",
        cfg.detector_weights,
        cfg.captioner_weights,
        cfg.captioner_backend,
        device,
    )

    detector = UltralyticsYoloDetector(cfg.detector_weights)
    captioner = load_caption_model(
        model_name=cfg.captioner_backend,
        model_name_or_path=str(cfg.captioner_weights),
        device=device,
    )
    return Omniparser(
        detector=detector,
        ocr_backend=EasyOcrBackend(),
        captioner=captioner,
        config=OmniparserConfig(),
    )


def _format_elements(elements: list[Any]) -> str:
    """Render the parsed-element list as a single human-readable string."""
    lines: list[str] = []
    for idx, el in enumerate(elements):
        kind = el["type"].capitalize()
        content = el["content"] if el["content"] is not None else "<no caption>"
        lines.append(f"{idx}. [{kind}] {content}")
    return "\n".join(lines) if lines else "<no elements detected>"


def _decode_annotated(annotated_b64: str) -> PILImage:
    import base64

    from PIL import Image as _Image

    return _Image.open(io.BytesIO(base64.b64decode(annotated_b64)))


def _process(
    slot: _ParserSlot,
    cfg: DemoConfig,
    image: PILImage | None,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
) -> tuple[PILImage | None, str]:
    if image is None:
        return None, "<upload an image first>"

    parser = slot.get(cfg)

    import dataclasses

    from omniparser.pipeline import parse_screen

    overridden = dataclasses.replace(
        parser.config,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        imgsz=(imgsz, imgsz),
    )
    result = parse_screen(
        image,
        detector=parser.detector,
        ocr_backend=parser.ocr_backend,
        captioner_bundle=parser.captioner,
        config=overridden,
    )
    annotated = _decode_annotated(result.annotated_image_b64)
    text = _format_elements(result.elements)
    _logger.info("demo_parsed element_count=%d", len(result.elements))
    return annotated, text


def build_demo(cfg: DemoConfig) -> gr.Blocks:
    """Build the Gradio :class:`Blocks` interface.

    Pure factory — no weights are loaded here, no server is started here.
    Tests can call this without paying for either.
    """
    import gradio as gr

    slot = _ParserSlot()

    with gr.Blocks(title="OmniParser demo") as demo:
        gr.Markdown(_MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Upload screenshot")
                box_th = gr.Slider(0.01, 1.0, value=0.05, step=0.01, label="Box threshold")
                iou_th = gr.Slider(0.01, 1.0, value=0.10, step=0.01, label="IoU threshold")
                use_paddle = gr.Checkbox(value=False, label="Use PaddleOCR")
                imgsz = gr.Slider(640, 1920, value=640, step=32, label="Detection image size")
                submit = gr.Button("Parse", variant="primary")
            with gr.Column():
                image_out = gr.Image(type="pil", label="Annotated output")
                text_out = gr.Textbox(label="Parsed elements", lines=18)

        submit.click(
            fn=lambda im, bt, it, up, sz: _process(slot, cfg, im, bt, it, up, sz),
            inputs=[image_in, box_th, iou_th, use_paddle, imgsz],
            outputs=[image_out, text_out],
        )

    return demo  # type: ignore[no-any-return]


def launch(cfg: DemoConfig) -> None:
    """Build and launch the demo. ``share`` defaults to False (loopback only)."""
    demo = build_demo(cfg)
    _logger.info(
        "demo_launching host=%s port=%d share=%s",
        cfg.server_name,
        cfg.server_port,
        cfg.share,
    )
    demo.launch(
        server_name=cfg.server_name,
        server_port=cfg.server_port,
        share=cfg.share,
        show_error=True,
    )
