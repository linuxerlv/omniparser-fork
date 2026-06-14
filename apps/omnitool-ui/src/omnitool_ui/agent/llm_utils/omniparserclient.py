"""Legacy ``OmniParserClient`` shim that delegates to ``omnitool-agent``.

Upstream's original ``OmniParserClient`` was a thin wrapper around raw
``requests.post(url, json={"base64_image": ...})``. We preserve the public
shape (class name, no-arg ``__call__`` that captures a screenshot, dict-like
return value with ``screen_info`` etc.) so the agent loop in
:mod:`omnitool_ui.loop` keeps working unchanged, while routing all wire
traffic through :class:`omnitool_agent.OmnitoolParserClient` (validated
schemas, structured error envelope, version pinning).

The screenshot is captured the same way the upstream did: by polling the
omnibox VM's ``/screenshot`` HTTP endpoint via
:func:`omnitool_ui.tools.screen_capture.get_screenshot`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from omnitool_agent import OmnitoolParserClient
from omnitool_ui.agent.llm_utils.utils import encode_image
from omnitool_ui.tools.screen_capture import get_screenshot

OUTPUT_DIR = "./tmp/outputs"


class OmniParserClient:
    """Drop-in replacement for upstream ``OmniParserClient``.

    The instance is callable: ``client()`` captures a screenshot and returns
    the legacy dict (``parsed_content_list``, ``som_image_base64``,
    ``screen_info``, ``latency``, ...). The agent loop relies on this
    no-argument ``__call__`` interface.
    """

    def __init__(self, url: str) -> None:
        # Upstream URLs were like ``http://host:8000/parse/`` (with trailing
        # slash). The new server exposes ``/parse``. We strip ``/parse`` /
        # ``/parse/`` if present so the user can pass either.
        base = url.rstrip("/")
        base = base.removesuffix("/parse")
        self._adapter = OmnitoolParserClient(base)

    def __call__(self) -> dict[str, Any]:
        screenshot, screenshot_path = get_screenshot()
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)

        result = self._adapter.parse_screenshot(image_base64)
        # Persist the SoM image alongside the screenshot, matching the
        # upstream behavior where ``screenshot_som_<uuid>.png`` was a
        # documented artifact.
        som_image_data = base64.b64decode(result["som_image_base64"])
        screenshot_uuid = Path(screenshot_path).stem.replace("screenshot_", "")
        som_screenshot_path = f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png"
        with open(som_screenshot_path, "wb") as f:
            f.write(som_image_data)

        # Override the auto-generated uuid with the screenshot's stem so
        # downstream callers can correlate the SoM file with the source PNG.
        return dict(result) | {
            "width": screenshot.size[0],
            "height": screenshot.size[1],
            "screenshot_uuid": screenshot_uuid,
        }
