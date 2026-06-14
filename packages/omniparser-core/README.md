# omniparser-core

Pure inference library: YOLO icon detection, Florence-2 captioning, OCR
adapters, IoU deduplication, and bounding-box annotation. The companion
HTTP service lives in `omniparser-server`; the lightweight HTTP client
that talks to it lives in `omniparser-client`.

This package is the migration target for `upstream/util/utils.py`,
`upstream/util/box_annotator.py`, and `upstream/util/omniparser.py`. It
is currently a skeleton; see `docs/migration-plan.md` at the workspace
root for the function-level migration table and acceptance criteria.

## Status

`0.1.0.dev0` — skeleton only, no functional code yet.

## Planned public API

```python
from omniparser import Omniparser, OmniparserConfig, ParseResult

parser = Omniparser(OmniparserConfig.from_huggingface("microsoft/OmniParser-v2.0"))
result: ParseResult = parser.parse(image)
for element in result.elements:
    print(element.bbox, element.content, element.interactivity)
```

## License

MIT (see `LICENSE` at the workspace root). Note that model weights loaded
at runtime carry their own licenses; `WEIGHTS_LICENSE.md` at the
workspace root documents the AGPL-3.0 implications of the YOLO-derived
icon detector weights.
