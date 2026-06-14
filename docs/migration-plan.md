# Migration Plan: OmniParser → Monorepo

This is the source of truth for migrating microsoft/OmniParser
(`b0d5c9f5701f7e2be4771872e6e928da77759df3`, v.2.0.1) into this monorepo.

The plan is divided into five phases. Each phase is independently
reviewable and produces a working state. P1-P4 may proceed in parallel
across different developers if needed.

---

## P0 — Foundation (configuration only, no code migration)

**Goal:** workspace resolves with `uv sync`, pre-commit runs clean on all
existing files, no source code is migrated yet.

**Deliverables**
- Root `pyproject.toml` with `[tool.uv.workspace]` declaring 7 members
- `LICENSE` (MIT), `NOTICE` (upstream attribution), `WEIGHTS_LICENSE.md`
  (AGPL exposure guidance for downstream commercial users)
- `ruff.toml` (shared lint), pyproject `[tool.mypy]` strict, `[tool.pytest]`
  with `--import-mode=importlib`, `[tool.coverage]`, `[tool.towncrier]`
- `.pre-commit-config.yaml` with ruff, mypy, nbstripout, gitleaks,
  check-added-large-files, conventional-pre-commit
- `.python-version`, completed `.gitignore`
- Per-package skeleton: empty `pyproject.toml` + `src/<pkg>/__init__.py`
  with `py.typed` marker

**Acceptance**
- `uv sync --all-packages --dev` succeeds
- `uv run ruff check .` returns 0 errors
- `uv run pre-commit run --all-files` returns 0 errors

---

## P1 — `omniparser-core` rewrite (the heaviest phase)

**Source:** `upstream/util/utils.py` (540 LOC), `upstream/util/box_annotator.py`
(262 LOC), `upstream/util/omniparser.py` (32 LOC).

### Design decisions (locked, 2026-06-14)

1. **Plugin backends.** `omniparser-core` ships with `Detector` and
   `OcrBackend` Protocols. Concrete implementations
   (`UltralyticsYoloDetector`, `EasyOcrBackend`, `PaddleOcrBackend`) live
   in submodules guarded by optional extras:
   - `pip install omniparser-core` — zero ML deps, Protocol stubs only
   - `pip install omniparser-core[yolo]` — adds `ultralytics` (AGPL-3.0)
   - `pip install omniparser-core[easyocr]` — adds `easyocr`
   - `pip install omniparser-core[paddleocr]` — adds `paddleocr`
   - `pip install omniparser-core[all]` — everything
   This isolates the AGPL surface to users who explicitly opt in, and
   makes core importable in CI without GPU/torch dependencies.

2. **Breaking API rewrite.** No back-compat shim for upstream's
   `get_som_labeled_img(image_source, ...)` 13-arg signature. The new
   surface is a single `Omniparser` class taking an `OmniparserConfig`,
   and a single `parse(image: Image | ndarray | Path) -> ParseResult`
   method. `BOX_TRESHOLD` typo is fixed without a kwarg alias because
   we're at `0.1.0.dev0`, not 1.0.

3. **`box_annotator.py` and `omniparser.py` migrate together** with
   `utils.py` in P1. P1 produces a complete, end-to-end-runnable
   `omniparser-core` package; P2 (server/client/demo) can then import
   it without further core work.

**Target tree:**
```
packages/omniparser-core/src/omniparser/
├── __init__.py             # re-export Omniparser, ParseResult, OmniparserConfig
├── api.py                  # Omniparser facade class (replaces util/omniparser.py)
├── detection.py            # YOLO loader and predict
├── captioning.py           # Florence-2 / BLIP-2 / Phi-3-V loaders and batch caption
├── ocr.py                  # OcrBackend protocol + EasyOcrBackend, PaddleOcrBackend
├── geometry.py             # box_area, intersection_area, IoU, is_inside,
│                           # int_box_area, remove_overlap_new
├── annotation.py           # BoxAnnotator + get_optimal_label_pos + annotate
├── pipeline.py             # parse_screen orchestration (no model state)
└── py.typed                # PEP 561 type marker
```

### Function-level migration table

| Source | Lines | Target file | Notes |
|---|---|---|---|
| `get_caption_model_processor` | utils.py:47-68 | `captioning.py` | Add `@lru_cache(maxsize=4)` for processor reuse |
| `get_yolo_model` | utils.py:71-75 | `detection.py` | Add type hint `-> YOLO` |
| `get_parsed_content_icon` | utils.py:78-122 | `captioning.py` | Replace 5 `print()` with `logger.debug` |
| `get_parsed_content_icon_phi3v` | utils.py:126-176 | `captioning.py` | Mark `# legacy`; gated by config |
| `remove_overlap` (old) | utils.py:178-228 | **DELETE** | Dead, only `remove_overlap_new` is called |
| `remove_overlap_new` | utils.py:231-309 | `pipeline.py` (orchestration part) + `geometry.py` (IoU helpers) | Split mixed concerns |
| `load_image` | utils.py:312-323 | **DELETE** | Dead (GroundingDINO leftover, 0 callers) |
| `annotate` | utils.py:326-354 | `annotation.py` | |
| `predict` | utils.py:357-375 | **DELETE** | Dead (HF GroundingDINO path, 0 callers) |
| `predict_yolo` | utils.py:378-399 | `detection.py` | |
| `int_box_area` | utils.py:401-405 | `geometry.py` | |
| `get_som_labeled_img` | utils.py:407-486 | `pipeline.py` as `parse_screen` | Reduce 13-arg signature via `OmniparserConfig` |
| `get_xywh` | utils.py:489-492 | `ocr.py` | |
| `get_xyxy` | utils.py:494-497 | `ocr.py` | |
| `get_xywh_yolo` | utils.py:499-502 | **DELETE** | Dead (0 callers) |
| `check_ocr_box` | utils.py:504-540 | `ocr.py` | Refactor `use_paddleocr` boolean → `OcrBackend` injection; remove `display_img` matplotlib branch (move to demo) |
| `BoxAnnotator` (class) | box_annotator.py:10-163 | `annotation.py` | |
| `get_optimal_label_pos` | box_annotator.py:189-261 | `annotation.py` | |
| `box_area`, `intersection_area`, `IoU` | box_annotator.py:165-186 | `geometry.py` | **Deduplicate** with the same functions nested inside `utils.py:remove_overlap` |
| `Omniparser` class | omniparser.py:7-32 | `api.py` | Rewrite: accept `Image | ndarray | Path`; base64 stays in server layer; remove `print()` |

### Mandatory patches (apply during migration)

| ID | Description | Locations | Impact |
|---|---|---|---|
| **A** | Eliminate import-time side effects | `utils.py:22-31` (`reader = easyocr.Reader(...)`, `paddle_ocr = PaddleOCR(...)`) | Replace with `OcrBackend.lazy_init()`; allows tests to import without 500MB model load |
| **B** | Device auto-detection | `utils.py:49`, `omniparser.py:10`, `gradio_demo.py:30`, `app_new.py` | `cuda → mps → cpu` fallback |
| **C** | Replace `print()` with `logging` | 88 occurrences across repo | Single `_get_logger(__name__)` helper |
| **D** | Externalize hardcoded paths | `gradio_demo.py:15-16` (`weights/icon_detect/model.pt`, `weights/icon_caption_florence`) | `huggingface_hub.snapshot_download` + env var override |
| **E** | Replace 5 bare `except:` | `utils.py:94`, `utils.py:294`, `vlm_agent.py:166`, `vlm_agent_with_orchestrator.py:229`, `eval/ss_pro_gpt4o_omniv2.py:202` | Specific exception classes, structured logging |
| **F** | Fix `BOX_TRESHOLD` typo | `utils.py:407`, `omniparser.py:30`, `gradio_demo.py:54` | Rename to `box_threshold` (kwargs back-compat alias for one minor version) |
| **G** | Replace `class Omniparser(object):` | `omniparser.py:7` | Drop Python-2-style explicit `object` |

### Tests required for P1

- `tests/test_geometry.py`: pure-function tests for IoU, is_inside (no models)
- `tests/test_pipeline.py`: smoke test using `MagicMock` detector + captioner +
  `OcrBackend` to exercise `parse_screen` orchestration without weights
- `tests/test_api.py`: integration smoke test gated by
  `@pytest.mark.slow` — downloads weights via huggingface_hub and parses
  one bundled sample screenshot

**Acceptance**
- `uv run pytest packages/omniparser-core` passes; coverage ≥ 60%
- `uv run mypy packages/omniparser-core/src` passes strict
- `uv run python -c "import omniparser; print(omniparser.__version__)"`
  completes in < 500 ms (proves no import-time side effects remain)

---

## P2 — `omniparser-server`, `omniparser-client`, `parser-demo`

**Goal:** the parser is reachable via HTTP from another process and from a
local Gradio demo.

**Sources & targets**
- `upstream/omnitool/omniparserserver/omniparserserver.py` (52 LOC)
  → `packages/omniparser-server/src/omniparser_server/{app.py,routes.py,settings.py}`
  - Remove `sys.path.append(root_dir)` hack (no longer needed once core is
    a real package)
  - Settings via `pydantic-settings` instead of CLI argparse
  - `uvicorn.run(..., reload=True)` is dev-only; production entry uses gunicorn
- `upstream/omnitool/gradio/agent/llm_utils/omniparserclient.py` (40 LOC)
  → `packages/omniparser-client/src/omniparser_client/__init__.py`
  - Standalone tiny package (only `requests` dependency; no torch)
  - Used by `omnitool-agent` and any third-party caller
- `upstream/gradio_demo.py` (96 LOC)
  → `apps/parser-demo/src/parser_demo/main.py`
  - Lazy load models inside `if __name__ == "__main__"`
  - `share=True` becomes opt-in via `--share` flag (was on by default)
  - Drop the `# import pdb; pdb.set_trace()` debug leftover

**Acceptance**
- `uv run --package omniparser-server uvicorn ...` boots, `/probe/` returns 200
- `omniparser-client` round-trips a sample image
- `apps/parser-demo` Gradio UI renders, parses one image

---

## P3 — `omnitool-agent`, `omnitool-ui`

**Goal:** the computer-use agent loop is a real Python package, with the UI
reduced to a single Gradio entry point.

**Sources & targets**
- `upstream/omnitool/gradio/loop.py` (117 LOC) → `omnitool-agent/loop.py`
- `upstream/omnitool/gradio/agent/{vlm_agent.py, vlm_agent_with_orchestrator.py, anthropic_agent.py}`
  → `omnitool-agent/agents/`
- `upstream/omnitool/gradio/agent/llm_utils/{oaiclient.py, groqclient.py, utils.py}`
  → `omnitool-agent/llm_clients/`
- `upstream/omnitool/gradio/executor/anthropic_executor.py` (123 LOC)
  → `omnitool-agent/executors/anthropic.py`
- `upstream/omnitool/gradio/tools/{base.py, collection.py, computer.py, screen_capture.py}`
  → `omnitool-agent/tools/`
- `upstream/omnitool/gradio/app_new.py` (707 LOC) → `apps/omnitool-ui/src/omnitool_ui/app.py`
  (rename, drop `_new` suffix)

**Deletions (during P3)**
- `upstream/omnitool/gradio/app.py` — superseded by `app_new.py`
- `upstream/omnitool/gradio/app_streamlit.py` — duplicate UI, no maintenance ROI

**Mandatory patches (apply during migration)**
- All bare `except:` blocks → typed exceptions (locations listed in P1 patch E)
- All `print()` → `logging` (covered by patch C)

**Acceptance**
- `uv run --package omnitool-ui python -m omnitool_ui` launches Gradio UI
- `uv run mypy packages/omnitool-agent/src` passes
  (apps/* is allowed to be relaxed)

---

## P4 — `omniparser-eval`, infra, CI/CD, docs, release

### `omniparser-eval`
- `upstream/eval/ss_pro_gpt4o_omniv2.py` (368 LOC)
  → `packages/omniparser-eval/src/omniparser_eval/screenspot_pro.py`
- **Fix broken import** `from models.utils import ...` (line 21) — this
  references a `models/` package that does not exist in upstream; rewrite
  to `from omniparser import ...`
- `upstream/eval/logs_sspro_omniv2.json` (1.16 MB) — does NOT migrate;
  publish as a GitHub release artifact instead

### Infra
- `upstream/omnitool/omnibox/` → `infra/omnibox/` (excluded from workspace)
- `upstream/omnitool/omnibox/vm/win11setup/setupscripts/server/main.py`
  → `infra/omnibox/agent-server/main.py` (not a workspace member)

### Notebooks & assets
- `upstream/demo.ipynb` → `examples/notebooks/quickstart.ipynb`
  (run `nbstripout` first; expect 287KB → ~30KB)
- Selected `imgs/*.png` smaller than 1MB → `docs/assets/`
- Larger demo images → git-lfs or release artifacts

### CI/CD workflows (`.github/workflows/`)
- `ci.yml`: lint + typecheck + test matrix (py3.11/3.12/3.13 ×
  ubuntu/macos/windows) + lockfile drift gate
- `release.yml`: triggered on `v*` tag; `uv build` + Trusted Publishing
  to PyPI (id-token: write, attestations: write)
- `security.yml`: weekly `pip-audit` and `cyclonedx-py environment` SBOM
- `docs.yml`: `mkdocs gh-deploy` on main

### Documentation site (`docs/mkdocs.yml`)
- mkdocs-material with `mkdocstrings-python`
- Auto-generate API docs from `omniparser-core` and `omniparser-client`
- Hand-written guides: install, quickstart, deploy as service,
  AGPL compliance

### Release plumbing
- Conventional Commits enforced via `conventional-pre-commit` (already in
  `.pre-commit-config.yaml`)
- `towncrier` configured at workspace root; news fragments in `changes/`
- Per-package SemVer: `omniparser-core 0.1.0`, others 0.1.0 initially
- First public release after P4 acceptance: `omniparser-core 0.1.0` to PyPI

**Acceptance**
- All three workflows pass on a representative PR
- `mkdocs serve` renders the API of `omniparser-core` without errors
- `uv build --package omniparser-core` produces a wheel installable
  in a clean venv from the local file path
- Trusted Publishing config registered on PyPI (manual one-time step,
  documented)

---

## Out of scope (intentional)

- Training pipeline. OmniParser does not include training code; this
  monorepo follows suit.
- License substitution for the YOLO weights (e.g. retraining a
  permissively-licensed icon detector). That is a project for downstream
  consumers; we document the AGPL implications in `WEIGHTS_LICENSE.md`
  and leave the choice to them.
- Migration of the omnibox Win11 VM image build. The Dockerfile and
  PowerShell setup scripts are preserved but stay in `infra/omnibox/`
  and are not part of the Python build, lint, or test surface.
