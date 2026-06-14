# Getting started

This guide walks you through cloning, installing, and running the parser end-to-end.

## Prerequisites

- **Python**: `>=3.11,<3.13`
- **uv**: [astral-sh/uv](https://docs.astral.sh/uv/) `>=0.5` (managed by `setup-uv` in CI; install locally via `pipx install uv` or the official installer)
- **git** with submodule support (the upstream snapshot lives in `upstream/`)

## Clone

```bash
git clone https://github.com/linuxerlv/omniparser-fork.git
cd omniparser-fork
git submodule update --init --recursive  # optional: only needed for parity tests
```

The `upstream/` submodule is **pinned** to commit `b0d5c9f5` (OmniParser v2.0.1) and is excluded from the build. It exists for traceability and a small set of contract tests; you can skip the submodule entirely for normal use.

## Install

```bash
uv sync --all-packages --group dev
```

This creates a `.venv/`, resolves the lockfile, and installs every workspace package plus the `dev` group (ruff, mypy, pytest, etc.). For a documentation build add `--group docs`; for benchmarks add `--group benchmarks`.

!!! tip "Frozen installs"
    CI uses `uv sync --frozen --all-packages --group dev`. Run the same command locally to catch lockfile drift before pushing.

## Verify the install

```bash
uv run ruff check
uv run ruff format --check
uv run mypy packages/omniparser-core/src \
              packages/omniparser-client/src \
              packages/omniparser-server/src \
              packages/omniparser-eval/src \
              packages/omnitool-agent/src
uv run pytest
```

A clean checkout should emit zero ruff issues, zero mypy errors, and **132 passed, 1 skipped** under `pytest` (the skipped test depends on the `upstream/` submodule contents).

## Parse a screenshot

The `omniparser-core` package provides a high-level `Omniparser` class. Weights are downloaded the first time you instantiate it:

```python
from pathlib import Path
from omniparser import Omniparser, OmniparserConfig

cfg = OmniparserConfig(
    box_threshold=0.05,
    iou_threshold=0.7,
    use_paddleocr=False,        # easyocr by default; PaddleOCR optional
    imgsz=640,
)
parser = Omniparser(cfg)        # downloads weights to HF cache on first call

result = parser.parse(Path("screenshot.png"))
print(result.elements[0])       # {'type': 'icon', 'bbox': [...], 'content': '...'}
print(result.label_coordinates) # dict[label_id, (x, y, w, h)] for visual debugging
```

`ParseResult` exposes:

- `annotated_image_b64: str` — base64 PNG with bounding boxes drawn
- `label_coordinates: dict[str, tuple[float, float, float, float]]` — `(x, y, w, h)` per label id
- `elements: list[ParsedElement]` — typed records: `{type, bbox, interactivity, content, source}`

### Use the Hugging Face mirror

Set `HF_ENDPOINT` before instantiating the parser if upstream Hugging Face is slow:

```bash
# bash / zsh
export HF_ENDPOINT=https://hf-mirror.com

# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

Both `omniparser-core` and `omniparser-eval` honor this variable.

## Run a grounding benchmark

The `omniparser-eval` package implements the **ScreenSpot-Pro** GPT-4X grounding evaluator from the upstream paper. The CLI runs the two methods that don't require local weights:

```bash
export OPENAI_API_KEY=sk-...
uv run omniparser-eval \
    --output runs/sspro.jsonl \
    --method allow_negative \
    --max-rows 20 \
    -v
```

Available methods:

| Method | Needs parser? | Description |
|---|---|---|
| `allow_negative` | no | Single-shot prompt; allows "Target not existent" answer |
| `with_uncertainty` | no | Two-stage prompt with uncertainty acknowledgment |
| `positive` | **yes** | Pre-parses the screenshot, asks the model to pick a label id (Python API only) |

For `positive`, configure an `Omniparser` instance and call `omniparser_eval.run_eval(...)` directly — see the API reference for `OpenAIGroundModel.ground_only_positive`.

## Next steps

- Read [Architecture](architecture.md) for the package layout and dependency graph.
- For development workflow (commits, releases, PR conventions), see the [internal migration plan](internal/migration-plan.md).
