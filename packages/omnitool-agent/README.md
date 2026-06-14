# omnitool-agent

Thin adapter layer that bridges `omniparser-client` (typed pydantic models)
into the legacy `omnitool` dict shape used by the original `OmniParserClient`
(`som_image_base64`, `parsed_content_list`, `screen_info`, `latency`).

This package is the *only* place in the monorepo that knows about the legacy
shape. Downstream UI code (e.g. `omnitool-ui`) depends on this adapter
instead of speaking HTTP directly, so swapping the wire format is a single
file change.

## Why a separate package

* `omnitool-ui` does not need to pull `httpx`, `pydantic`, or the wire
  schemas directly — it only needs the adapter result.
* The agent loop's expectations were pinned by upstream's `OmniParserClient`
  in `agent/llm_utils/omniparserclient.py`. We preserve that contract while
  routing all traffic through our typed `OmniparserClient`.
* Allows independent testing of the shape mapping without spinning up Gradio.

## Public surface

```python
from omnitool_agent import OmnitoolParserClient, ParsedScreen

client = OmnitoolParserClient(base_url="http://localhost:8000")
parsed: ParsedScreen = client.parse_screenshot(image_b64)
parsed["parsed_content_list"]  # list of legacy element dicts
parsed["som_image_base64"]      # annotated PNG, base64
parsed["screen_info"]           # human-readable summary string
parsed["latency"]               # seconds (float)
```
