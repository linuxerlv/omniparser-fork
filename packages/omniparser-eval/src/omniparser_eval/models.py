"""Grounding model backends.

This module ports the four grounding methods from upstream
``upstream/eval/ss_pro_gpt4o_omniv2.py`` ``GPT4XModel`` (lines 84-382)
behind a typed :class:`GroundModel` ``Protocol`` so eval drivers can be
written against the abstraction and tested without OpenAI access.

Differences vs. upstream:

* **Schema**: ``GroundResult.bbox`` and ``GroundResult.point`` are
  **absolute pixel** integers (xyxy / xy), not normalized [0,1] floats.
  This matches the JSONL schema in ``upstream/eval/logs_sspro_omniv2.json``
  exactly (e.g. ``"pred": [1396, 1215]``, ``"bbox": [2042, 1153, 2067, 1182]``).
* **DI**: model construction takes ``client`` (``openai.OpenAI``) and
  ``parser`` (``omniparser.Omniparser``) by injection. No module-level
  globals, no ``OPENAI_API_KEY`` reads at import time, no path placeholders
  like upstream ``SOM_MODEL_PATH='...'`` (literally an ellipsis string,
  upstream lines 26-28).
* **Generation config**: upstream references ``self.override_generation_config``
  in lines 174, 259, 343 but never initializes it in ``__init__`` (upstream
  bug). We initialize it to ``{"temperature": 0.0}`` and provide
  :meth:`set_generation_config` to override.
* **phi35v removed**: upstream ``ground_only_positive_phi35v`` (lines 99-124)
  depends on ``get_pred_phi3v`` and ``get_phi3v_model_dict`` symbols that
  do not exist anywhere in upstream's published codebase, so it is
  unreachable. Removed per ADR D11.
* **Prompt templates**: copied **verbatim** from upstream (lines 71, 147,
  236, 250-254, 317, 331-337) for byte-for-byte parity with the published
  benchmark log. Modifications would invalidate the benchmark numbers.

Parity note (upstream bug preserved): in :meth:`OpenAIGroundModel.ground_only_positive`
we keep the upstream click-point formula
``click_point = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]`` (upstream
line 201). This is mathematically ``[x + w/2, y + h/2]`` which is the
**bbox center** assuming ``label_coordinates[label] == (x, y, w, h)``. The
``omniparser-core`` ``ParseResult.label_coordinates`` happens to use the
same ``(x, y, w, h)`` convention, so the formula is correct in both
codebases — but it's worth flagging because the variable name ``bbox``
suggests ``(x1, y1, x2, y2)`` which would be wrong.
"""

from __future__ import annotations

import ast
import base64
import io
import logging
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast, runtime_checkable

if TYPE_CHECKING:
    import openai
    from PIL import Image

    from omniparser import Omniparser
    from omniparser.pipeline import ParsedElement

__all__ = [
    "GroundModel",
    "GroundResult",
    "OpenAIGroundModel",
    "denormalize_coords",
    "extract_first_bounding_box",
    "extract_first_point",
    "reformat_messages",
]

_logger = logging.getLogger(__name__)

# Upstream prompt templates — copied verbatim from
# upstream/eval/ss_pro_gpt4o_omniv2.py for benchmark parity.
# DO NOT MODIFY without coordinating a benchmark re-run.

_PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_V1 = (
    "The instruction is to {}. \n"
    "Here is the list of all detected bounding boxes by IDs and their descriptions: {}. \n"
    "Keep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n"
    " Requirement: 1. You should first give a reasonable description of the current screenshot, "
    "and give a some analysis of how can the user instruction be achieved by a single click. "
    "2. Then make an educated guess of bbox id to click in order to complete the task using both "
    "the visual information from the screenshot image and the bounding boxes descriptions. "
    "REMEMBER: the task instruction must be achieved by one single click. "
    "3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. "
    "Please do not include any other info."
)

_SYSTEM_PROMPT_GROUND_ONLY_POSITIVE = (
    "You are an expert at completing instructions on GUI screens. \n"
    "               You will be presented with two images. The first is the original screenshot. "
    "The second is the same screenshot with some numeric tags. You will also be provided with "
    "some descriptions of the bbox, and your task is to choose the numeric bbox idx you want to "
    "click in order to complete the user instruction."
)

_SYSTEM_PROMPT_NEGATIVE_OR_UNCERTAINTY = (
    "You are an expert in using electronic devices and interacting with graphic interfaces. "
    "You should not call any external tools."
)

_USER_PROMPT_GROUND_ALLOW_NEGATIVE = (
    "You are asked to find the bounding box of an UI element in the given screenshot "
    "corresponding to a given instruction.\n"
    "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], "
    "with x and y ranging from 0 to 1. \n"
    "If such element does not exist, output only the text 'Target not existent'.\n"
    "The instruction is:\n"
    "{instruction}\n"
)

_USER_PROMPT_GROUND_WITH_UNCERTAINTY = (
    "You are asked to find the bounding box of an UI element in the given screenshot "
    "corresponding to a given instruction.\n"
    "- If such element does not exist in the screenshot, output only the text 'Target not existent'."
    "- If you are sure such element exists and you are confident in finding it, output your "
    "result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. \n"
    "Please find out the bounding box of the UI element corresponding to the following instruction: \n"
    "The instruction is:\n"
    "{instruction}\n"
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class GroundResult(TypedDict, total=False):
    """Return shape from any grounding call.

    Schema matches upstream ``logs_sspro_omniv2.json`` rows. All coordinate
    fields are in **absolute pixels** of the input image (origin top-left,
    integer-valued).

    Fields:

    * ``result``: ``"positive"`` | ``"negative"`` | ``"failed"``
    * ``bbox``: ``[x1, y1, x2, y2]`` absolute pixels, or ``None``
    * ``point``: ``[x, y]`` absolute pixels (click target), or ``None``
    * ``raw_response``: parsed dict (JSON-like) or string from the model.
      Upstream stored either form depending on whether ``ast.literal_eval``
      succeeded; we preserve that flexibility.
    * ``dino_labled_img``: optional base64 PNG of SoM-annotated image
    * ``screen_info``: optional HTML-ish string of element descriptions
    """

    result: Literal["positive", "negative", "failed"]
    bbox: list[int] | None
    point: list[int] | None
    raw_response: dict[str, Any] | str | None
    dino_labled_img: str | None
    screen_info: str | None


@runtime_checkable
class GroundModel(Protocol):
    """Protocol every grounding backend must satisfy.

    ``image`` is a PIL ``Image.Image`` (RGB). Implementations MUST return a
    :class:`GroundResult` with absolute-pixel coordinates relative to the
    input image's natural size.
    """

    def ground_only_positive(self, instruction: str, image: Image.Image) -> GroundResult: ...

    def ground_allow_negative(self, instruction: str, image: Image.Image) -> GroundResult: ...

    def ground_with_uncertainty(self, instruction: str, image: Image.Image) -> GroundResult: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert_pil_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def reformat_messages(parsed_content_list: list[ParsedElement]) -> str:
    """Render parsed elements as the HTML-ish description block sent to GPT-4o.

    Format ported from ``upstream/eval/ss_pro_gpt4o_omniv2.py`` lines 53-63.
    Each element is mutated to add an ``idx`` field (preserved from upstream
    behavior — eval callers depend on this side effect).
    """
    screen_info = ""
    for idx, element in enumerate(parsed_content_list):
        # Upstream mutates the element dict; we preserve that contract because
        # downstream code may consult ``element["idx"]``. Cast to plain dict
        # since ParsedElement is a TypedDict without an "idx" key.
        cast("dict[str, Any]", element)["idx"] = idx
        if element["type"] == "text":
            screen_info += f'<p id={idx} class="text" alt="{element["content"]}"> </p>\n'
        elif element["type"] == "icon":
            screen_info += f'<img id={idx} class="icon" alt="{element["content"]}"> </img>\n'
    return screen_info


def extract_first_bounding_box(text: str) -> list[float] | None:
    """Find first ``[[x0,y0,x1,y1]]`` pattern in text. Ported from upstream lines 384-396."""
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        ]
    return None


def extract_first_point(text: str) -> list[float] | None:
    """Find first ``[[x0,y0]]`` pattern in text. Ported from upstream lines 399-410."""
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None


def _extract_dict_from_text(text: str) -> dict[str, Any]:
    """Extract the first ``{...}`` block from text and ``ast.literal_eval`` it.

    Ported behaviorally from upstream ``models.utils.extract_dict_from_text``
    which is referenced at line 204 but lives outside the published file. We
    implement the obvious behavior: find the first balanced brace block and
    parse it as a Python literal.
    """
    # Find first '{' and matching '}'.
    start = text.find("{")
    if start == -1:
        msg = "no dict literal found in text"
        raise ValueError(msg)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                result = ast.literal_eval(snippet)
                if not isinstance(result, dict):
                    msg = "extracted literal is not a dict"
                    raise ValueError(msg)
                return result
    msg = "unbalanced braces in text"
    raise ValueError(msg)


def denormalize_coords(
    coords: list[float],
    image_size: tuple[int, int],
) -> list[int]:
    """Map normalized [0,1] coords to absolute pixel ints.

    For a 2-tuple (point) returns ``[x*W, y*H]``.
    For a 4-tuple (bbox xyxy) returns ``[x1*W, y1*H, x2*W, y2*H]``.
    """
    width, height = image_size
    if len(coords) == 2:
        return [round(coords[0] * width), round(coords[1] * height)]
    if len(coords) == 4:
        return [
            round(coords[0] * width),
            round(coords[1] * height),
            round(coords[2] * width),
            round(coords[3] * height),
        ]
    msg = f"expected 2- or 4-tuple, got len={len(coords)}"
    raise ValueError(msg)


def _label_coord_to_xywh_int(
    raw: Any,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Coerce a single label_coordinates entry to absolute (x, y, w, h) ints.

    ``omniparser.ParseResult.label_coordinates`` may store entries as
    normalized floats (when ``output_coord_in_ratio=True``, the default) or
    absolute pixels. We probe by magnitude: if all four are <= 1.5 we treat
    as normalized. This matches upstream's implicit convention.
    """
    width, height = image_size
    x, y, w, h = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    is_normalized = all(abs(v) <= 1.5 for v in (x, y, w, h))
    if is_normalized:
        return (
            round(x * width),
            round(y * height),
            round(w * width),
            round(h * height),
        )
    return (round(x), round(y), round(w), round(h))


# ---------------------------------------------------------------------------
# OpenAIGroundModel
# ---------------------------------------------------------------------------


class OpenAIGroundModel:
    """OpenAI GPT-4o grounding backend (parity with upstream ``GPT4XModel``).

    Construction is fully explicit — the OpenAI client and the OmniParser
    parser are injected, no environment variable reads, no module-level
    state.

    Parity scope:

    * ``ground_only_positive`` matches upstream lines 126-218 (uses parsed
      content list + SoM image; GPT-4o picks a bbox ID).
    * ``ground_allow_negative`` matches upstream lines 220-298 (single
      screenshot; GPT-4o emits a normalized bbox or "Target not existent").
    * ``ground_with_uncertainty`` matches upstream lines 301-382 (single
      screenshot, prompt allows uncertainty escape hatch).
    * ``ground_only_positive_phi35v`` is intentionally **not** ported (D11).
    """

    def __init__(
        self,
        *,
        client: openai.OpenAI,
        parser: Omniparser | None = None,
        model_name: str = "gpt-4o-2024-05-13",
        max_tokens: int = 2048,
    ) -> None:
        """Inject the OpenAI client and (optionally) the OmniParser parser.

        ``parser`` is required only for :meth:`ground_only_positive`, which
        needs a SoM-annotated image and an element list. The other two
        methods send raw screenshots and don't need a parser.
        """
        self.client = client
        self.parser = parser
        self.model_name = model_name
        self.max_tokens = max_tokens
        # Upstream bug fix: this attribute is referenced in upstream lines
        # 174/259/343 but never initialized in __init__. We initialize it
        # explicitly.
        self.override_generation_config: dict[str, Any] = {"temperature": 0.0}

    def set_generation_config(self, **kwargs: Any) -> None:
        """Update generation config in-place. Mirrors upstream lines 96-97."""
        self.override_generation_config.update(kwargs)

    # ------------------------------------------------------------------
    # ground_only_positive
    # ------------------------------------------------------------------

    def ground_only_positive(
        self,
        instruction: str,
        image: Image.Image,
    ) -> GroundResult:
        """Ground a positive instruction using parsed elements + SoM image.

        Returns absolute-pixel ``bbox`` and ``point``.

        Parity: upstream lines 126-218.
        """
        from openai import BadRequestError

        if self.parser is None:
            msg = "ground_only_positive requires a parser; pass parser= to OpenAIGroundModel"
            raise ValueError(msg)

        image_size = image.size  # (W, H)
        base64_image = _convert_pil_image_to_base64(image)
        result = self.parser.parse(image)
        dino_labled_img = result.annotated_image_b64
        label_coordinates = result.label_coordinates
        screen_info = reformat_messages(result.elements)

        prompt_origin = _PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_V1.format(instruction, screen_info)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": _SYSTEM_PROMPT_GROUND_ONLY_POSITIVE},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_origin},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{dino_labled_img}"},
                            },
                        ],
                    },
                ],
                temperature=self.override_generation_config.get("temperature", 0.0),
                max_tokens=self.max_tokens,
            )
            response_text = response.choices[0].message.content or ""
        except BadRequestError:
            _logger.exception("OpenAI BadRequestError in ground_only_positive")
            return GroundResult(result="failed", bbox=None, point=None, raw_response=None)

        # Strip markdown code fences (upstream line 194).
        cleaned = response_text.replace("```json", "").replace("```", "")

        parsed: dict[str, Any]
        try:
            evaluated = ast.literal_eval(cleaned)
            if not isinstance(evaluated, dict):
                raise ValueError
            parsed = evaluated
        except (ValueError, SyntaxError):
            _logger.warning("ast.literal_eval failed, falling back to regex dict extractor")
            parsed = _extract_dict_from_text(cleaned)

        icon_id = str(parsed["Click BBox ID"])
        x, y, w, h = _label_coord_to_xywh_int(label_coordinates[icon_id], image_size)
        # Upstream click-point formula (line 201):
        # click_point = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        # which is bbox center for (x,y,w,h) layout. Preserved verbatim.
        click_point = [x + w // 2, y + h // 2]
        # Upstream stored "bbox" as the same (x,y,w,h) tuple. We convert to
        # xyxy here because downstream scoring expects xyxy.
        bbox_xyxy = [x, y, x + w, y + h]

        return GroundResult(
            result="positive",
            bbox=bbox_xyxy,
            point=click_point,
            raw_response=parsed,
            dino_labled_img=dino_labled_img,
            screen_info=screen_info,
        )

    # ------------------------------------------------------------------
    # ground_allow_negative
    # ------------------------------------------------------------------

    def ground_allow_negative(
        self,
        instruction: str,
        image: Image.Image,
    ) -> GroundResult:
        """Ground an instruction, allowing 'Target not existent'.

        Parity: upstream lines 220-298.
        """
        from openai import BadRequestError

        image_size = image.size
        base64_image = _convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": _SYSTEM_PROMPT_NEGATIVE_OR_UNCERTAINTY},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                            {
                                "type": "text",
                                "text": _USER_PROMPT_GROUND_ALLOW_NEGATIVE.format(
                                    instruction=instruction,
                                ),
                            },
                        ],
                    },
                ],
                temperature=self.override_generation_config.get("temperature", 0.0),
                max_tokens=self.max_tokens,
            )
            response_text = response.choices[0].message.content or ""
        except BadRequestError:
            _logger.exception("OpenAI BadRequestError in ground_allow_negative")
            return GroundResult(result="failed", bbox=None, point=None, raw_response=None)

        if "not existent" in response_text.lower():
            return GroundResult(
                result="negative",
                bbox=None,
                point=None,
                raw_response=response_text,
            )

        bbox_norm = extract_first_bounding_box(response_text)
        point_norm = extract_first_point(response_text)
        if bbox_norm and not point_norm:
            point_norm = [
                (bbox_norm[0] + bbox_norm[2]) / 2,
                (bbox_norm[1] + bbox_norm[3]) / 2,
            ]

        bbox_px = denormalize_coords(bbox_norm, image_size) if bbox_norm else None
        point_px = denormalize_coords(point_norm, image_size) if point_norm else None

        return GroundResult(
            result="positive" if (bbox_px or point_px) else "negative",
            bbox=bbox_px,
            point=point_px,
            raw_response=response_text,
        )

    # ------------------------------------------------------------------
    # ground_with_uncertainty
    # ------------------------------------------------------------------

    def ground_with_uncertainty(
        self,
        instruction: str,
        image: Image.Image,
    ) -> GroundResult:
        """Ground an instruction with uncertainty escape hatch.

        Parity: upstream lines 301-382.
        """
        from openai import BadRequestError

        image_size = image.size
        base64_image = _convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": _SYSTEM_PROMPT_NEGATIVE_OR_UNCERTAINTY},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                            {
                                "type": "text",
                                "text": _USER_PROMPT_GROUND_WITH_UNCERTAINTY.format(
                                    instruction=instruction,
                                ),
                            },
                        ],
                    },
                ],
                temperature=self.override_generation_config.get("temperature", 0.0),
                max_tokens=self.max_tokens,
            )
            response_text = response.choices[0].message.content or ""
        except BadRequestError:
            _logger.exception("OpenAI BadRequestError in ground_with_uncertainty")
            return GroundResult(result="failed", bbox=None, point=None, raw_response=None)

        # Upstream uses "not found" here (line 360); that's an upstream
        # inconsistency vs. the prompt which says "Target not existent".
        # We accept either to be robust.
        lower = response_text.lower()
        if "not existent" in lower or "not found" in lower:
            return GroundResult(
                result="negative",
                bbox=None,
                point=None,
                raw_response=response_text,
            )

        bbox_norm = extract_first_bounding_box(response_text)
        point_norm = extract_first_point(response_text)
        if bbox_norm and not point_norm:
            point_norm = [
                (bbox_norm[0] + bbox_norm[2]) / 2,
                (bbox_norm[1] + bbox_norm[3]) / 2,
            ]

        bbox_px = denormalize_coords(bbox_norm, image_size) if bbox_norm else None
        point_px = denormalize_coords(point_norm, image_size) if point_norm else None

        return GroundResult(
            result="positive",
            bbox=bbox_px,
            point=point_px,
            raw_response=response_text,
        )


# Compile-time assurance that the concrete class satisfies the Protocol.
_: type[GroundModel] = OpenAIGroundModel
