# OmniParser Monorepo

> Engineering-grade reorganization of [microsoft/OmniParser](https://github.com/microsoft/OmniParser) (upstream commit `b0d5c9f5`, v2.0.1) into a standards-conformant Python `uv` workspace monorepo.

This site documents the **rebuilt** packaging, APIs, and engineering standards. The upstream project ships excellent research code but lacks packaging, type checking, tests, and CI; this fork preserves the inference behavior bit-for-bit while wrapping it in installable, independently-versioned packages.

## Project status

!!! info "Pre-1.0"
    The package APIs are stabilizing. Public surface is `omniparser-core`, `omniparser-client`, and `omniparser-eval`. Other packages are reorganized but not yet published.

| Package | Status | Description |
|---|---|---|
| `omniparser-core` | stable interface | Pure inference library: YOLO detection, caption, OCR, IoU dedup |
| `omniparser-client` | stable interface | Typed HTTP client wrapping the FastAPI service |
| `omniparser-server` | stable interface | FastAPI service exposing `omniparser-core` |
| `omniparser-eval` | new | ScreenSpot-Pro grounding benchmarks (GPT-4X) |
| `omnitool-agent` | reorganized | VLM agent loop (consolidated from `upstream/omnitool/`) |

## What's different from upstream

- **Packaging**: PEP 621 `pyproject.toml`, `src/` layout, `uv.lock` for reproducible installs
- **Type checking**: `mypy --strict` on every package's `src/` tree
- **Linting**: `ruff` (replaces black + isort + flake8)
- **Tests**: `pytest` with `--import-mode=importlib`; mocked, no network or weights
- **CI**: GitHub Actions matrix (Linux/Windows/macOS × py3.11/3.12), pinned by SHA
- **Supply chain**: Dependabot weekly updates, CodeQL on push/PR/weekly, dependency review on PRs
- **Weights**: downloaded at runtime via `huggingface_hub.snapshot_download`; mirror-aware (`HF_ENDPOINT`)

## Licensing

The monorepo is **MIT** (`LICENSE`). Upstream OmniParser is **CC-BY-4.0**, retained verbatim under `upstream/LICENSE`. Model weights distributed by Microsoft (`microsoft/OmniParser-v2.0`) carry their own licenses — notably **AGPL-3.0** for the `icon_detect` YOLO weights. Weights are downloaded on demand and **never** redistributed in this repository. Downstream users are responsible for AGPL compliance when self-hosting weights.

## Where to next

- **[Getting started](getting-started.md)** — install, run a parse, run the eval harness
- **[Architecture](architecture.md)** — package boundaries, dependency graph, build system
- **[Migration plan (internal)](internal/migration-plan.md)** — the source-of-truth plan that drove this restructure
