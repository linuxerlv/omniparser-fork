"""Caption backends for omniparser-core.

Captioners turn cropped icon thumbnails into short descriptive strings.
Three model families are supported via the ``[caption]`` extra:

* Florence-2 (``microsoft/Florence-2-base``, MIT licensed)
* BLIP-2 (``Salesforce/blip2-opt-2.7b``, BSD-3-Clause)
* Phi-3-Vision (legacy upstream path, gated by config)

Loading is lazy: importing this module does not pull torch or
transformers, so :func:`omniparser.parse_screen` can run on CPU-only
machines without GPU drivers when callers supply their own captioner.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage

__all__ = [
    "CaptionModelBundle",
    "Captioner",
    "caption_icons",
    "caption_icons_phi3v",
    "load_caption_model",
]

_logger = logging.getLogger(__name__)


class CaptionModelBundle(Protocol):
    """Loose structural type for upstream's captioner dict.

    Upstream OmniParser passes ``{'model': ..., 'processor': ...}`` around as
    the captioner handle. Kept as a Protocol so callers that already have a
    HuggingFace pipeline can plug it in without inheriting any concrete class.
    """

    model: Any
    processor: Any


@runtime_checkable
class Captioner(Protocol):
    """Pluggable icon captioner.

    Implementations receive cropped 64x64 RGB icons and produce a list of
    short textual descriptions, one per icon, in input order.
    """

    def caption(
        self,
        crops: list[PILImage],
        *,
        prompt: str | None = None,
        batch_size: int = 128,
    ) -> list[str]:
        """Caption a list of cropped icon images."""
        ...


def load_caption_model(
    model_name: str,
    model_name_or_path: str = "Salesforce/blip2-opt-2.7b",
    device: str | None = None,
) -> dict[str, Any]:
    """Load a caption model and its processor.

    Returns the upstream ``{'model': ..., 'processor': ...}`` dict so that
    existing HuggingFace pipelines remain interoperable.

    ``device`` defaults to ``cuda`` if available, else ``mps`` on Apple
    Silicon, else ``cpu`` (patch B).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = (
            "load_caption_model requires the 'caption' extra. "
            "Install with: pip install 'omniparser-core[caption]'"
        )
        raise ImportError(msg) from exc

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    _logger.debug("Caption model device resolved to %s", device)

    if model_name == "blip2":
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=dtype
        )
        if device != "cpu":
            model = model.to(device)
    elif model_name == "florence2":
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, trust_remote_code=True
        )
        if device != "cpu":
            model = model.to(device)
    else:
        msg = f"Unsupported caption model name: {model_name!r} (expected 'blip2' or 'florence2')"
        raise ValueError(msg)

    return {"model": model.to(device), "processor": processor}


def _crop_icons(
    image: np.ndarray,
    boxes_normalised: list[list[float]] | Any,
    *,
    skip_first_n: int = 0,
    target_size: tuple[int, int] = (64, 64),
) -> list[PILImage]:
    """Crop icon thumbnails out of ``image`` according to normalised boxes.

    Skips boxes that fail to crop (e.g. zero-area after rounding) and
    logs the failure rather than swallowing it silently (patch E).
    """
    import cv2  # local import: only needed when captioning runs
    from torchvision.transforms import ToPILImage

    to_pil = ToPILImage()
    h, w = image.shape[:2]
    crops: list[PILImage] = []
    boxes = boxes_normalised[skip_first_n:] if skip_first_n else boxes_normalised
    for idx, coord in enumerate(boxes):
        try:
            xmin, xmax = int(coord[0] * w), int(coord[2] * w)
            ymin, ymax = int(coord[1] * h), int(coord[3] * h)
            cropped = image[ymin:ymax, xmin:xmax, :]
            cropped = cv2.resize(cropped, target_size)
            crops.append(to_pil(cropped))
        except (ValueError, cv2.error) as exc:
            _logger.debug("Failed to crop icon %d: %s", idx, exc)
            continue
    return crops


def caption_icons(
    boxes_normalised: list[list[float]] | Any,
    starting_idx: int,
    image: np.ndarray,
    bundle: dict[str, Any],
    *,
    prompt: str | None = None,
    batch_size: int = 128,
) -> list[str]:
    """Caption icons cropped from ``image`` using a Florence-2 / BLIP-2 model.

    Replaces upstream ``get_parsed_content_icon``. ``starting_idx`` is the
    offset into ``boxes_normalised`` at which the icon (non-OCR) boxes
    begin; everything before is OCR text and is skipped.
    """
    import torch

    crops = _crop_icons(image, boxes_normalised, skip_first_n=starting_idx)
    if not crops:
        return []

    model = bundle["model"]
    processor = bundle["processor"]
    device = model.device

    if prompt is None:
        prompt = "<CAPTION>" if "florence" in model.config.name_or_path else "The image shows"

    generated_texts: list[str] = []
    for offset in range(0, len(crops), batch_size):
        t0 = time.time()
        batch = crops[offset : offset + batch_size]
        if device.type == "cuda":
            inputs = processor(
                images=batch,
                text=[prompt] * len(batch),
                return_tensors="pt",
                do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(
                device=device
            )

        if "florence" in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,
            )
        generated = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts.extend(g.strip() for g in generated)
        _logger.debug(
            "Caption batch %d-%d (%d crops) in %.2fs",
            offset,
            offset + len(batch),
            len(batch),
            time.time() - t0,
        )

    return generated_texts


def caption_icons_phi3v(
    boxes_normalised: list[list[float]] | Any,
    ocr_bbox: list[Any] | None,
    image: np.ndarray,
    bundle: dict[str, Any],
) -> list[str]:
    """Caption icons using Phi-3-Vision.

    Preserved verbatim from upstream for compatibility, but marked as a
    legacy code path: Florence-2 is the default in v2.x. Gate access
    through :class:`OmniparserConfig` rather than calling this directly.
    """
    import torch

    skip = len(ocr_bbox) if ocr_bbox else 0
    crops = _crop_icons(image, boxes_normalised, skip_first_n=skip, target_size=(64, 64))
    if not crops:
        return []

    model = bundle["model"]
    processor = bundle["processor"]
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    batch_size = 5
    generated_texts: list[str] = []
    for offset in range(0, len(crops), batch_size):
        images = crops[offset : offset + batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "image_sizes": [],
        }
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            inp = processor._convert_images_texts_to_inputs(
                image_inputs[i], txt, return_tensors="pt"
            )
            inputs["input_ids"].append(inp["input_ids"])
            inputs["attention_mask"].append(inp["attention_mask"])
            inputs["pixel_values"].append(inp["pixel_values"])
            inputs["image_sizes"].append(inp["image_sizes"])
        max_len = max(x.shape[1] for x in inputs["input_ids"])
        for i, v in enumerate(inputs["input_ids"]):
            pad = processor.tokenizer.pad_token_id * torch.ones(
                1, max_len - v.shape[1], dtype=torch.long
            )
            inputs["input_ids"][i] = torch.cat([pad, v], dim=1)
            inputs["attention_mask"][i] = torch.cat(
                [
                    torch.zeros(1, max_len - v.shape[1], dtype=torch.long),
                    inputs["attention_mask"][i],
                ],
                dim=1,
            )
        cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        gen_args = {"max_new_tokens": 25, "temperature": 0.01, "do_sample": False}
        gen_ids = model.generate(**cat, eos_token_id=processor.tokenizer.eos_token_id, **gen_args)
        gen_ids = gen_ids[:, cat["input_ids"].shape[1] :]
        response = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generated_texts.extend(r.strip("\n").strip() for r in response)

    return generated_texts
