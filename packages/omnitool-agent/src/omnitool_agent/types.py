r"""Legacy dict shape used by the original ``omnitool`` UI / agent loop.

Upstream's ``OmniParserClient.__call__`` returned a plain ``dict[str, Any]``
shaped roughly like::

    {
        "som_image_base64": str,  # annotated PNG, base64 (no data: prefix)
        "parsed_content_list": list[dict],
        "latency": float,  # seconds
        "width": int,  # original screenshot width
        "height": int,  # original screenshot height
        "original_screenshot_base64": str,
        "screenshot_uuid": str,  # filename stem of source PNG
        "screen_info": str,  # human-readable, "ID: 0, Text: foo\\n..."
    }

We capture that contract as a :class:`typing.TypedDict` so downstream code
can keep typing turned on without rewriting the rest of the agent loop.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class LegacyParsedElement(TypedDict, total=False):
    """One element in ``parsed_content_list``.

    Marked ``total=False`` because upstream's loop tolerates missing keys
    (e.g. ``content`` may be absent when captioning is disabled). We always
    populate ``type``, ``bbox``, ``interactivity``, ``content``, ``source``,
    and ``idx`` from the adapter; legacy callers may add more.
    """

    type: Literal["text", "icon"]
    bbox: list[float]  # [x1, y1, x2, y2], normalized to [0, 1]
    interactivity: bool
    content: str | None
    source: str
    idx: int


class ParsedScreen(TypedDict):
    """Full payload returned by :class:`omnitool_agent.OmnitoolParserClient`.

    This is the shape upstream code expects from
    ``OmniParserClient()`` (i.e. ``response_json`` after
    ``reformat_messages``).
    """

    som_image_base64: str
    parsed_content_list: list[LegacyParsedElement]
    latency: float
    width: int
    height: int
    original_screenshot_base64: str
    screenshot_uuid: str
    screen_info: str
