# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) for
versioning. Pre-release tags (`aN`, `bN`, `rcN`, `.devN`) ship as GitHub
Releases only and are not published to PyPI.

This is a fork of [microsoft/OmniParser](https://github.com/microsoft/OmniParser)
restructured as a uv-managed Python monorepo. The upstream commit pinned by the
`upstream/` submodule is `b0d5c9f5` (release `v2.0.1`). All packages in this
workspace share a single version line.

## [Unreleased]

## [0.1.0a1] - 2026-06-14

First tagged pre-release. Establishes the monorepo, packaging story, CI/CD
pipeline, evaluation harness, and documentation site. No stable API guarantees
yet — every public symbol may change before `0.1.0`.

### Added

- **Monorepo layout** — uv workspace with seven members under `packages/` and
  `apps/`, all built with Hatchling and the `src/` layout: `omniparser-core`,
  `omniparser-server`, `omniparser-client`, `omniparser-eval`, `omnitool-agent`,
  `parser-demo`, `omnitool-ui`. Python `>=3.11,<3.13`.
- **`omniparser-core`** — typed port of the upstream icon-detection and OCR
  pipeline. Lazy weight loading via `huggingface_hub.snapshot_download`,
  defaulting to `HF_ENDPOINT=https://hf-mirror.com` so checkpoints stay out of
  the repo.
- **`omniparser-server`** — FastAPI service exposing the core pipeline with a
  typed request/response schema and structured logging.
- **`omniparser-client`** — async HTTP client with retries, generated from the
  server's OpenAPI schema.
- **`omniparser-eval`** — port of the upstream GPT-4X grounding benchmark.
  Replays `upstream/eval/logs_sspro_omniv2.json` through a typed pipeline and
  emits machine-readable scorecards. Network and weight access are gated behind
  `@pytest.mark.slow`; the default suite uses fixtures.
- **`omnitool-agent`** — typed agent loop wrapping the upstream `omnitool/`
  reference implementation.
- **`apps/parser-demo`** and **`apps/omnitool-ui`** — Gradio launchers for
  manual smoke-testing.
- **CI** — `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`,
  `mypy --strict`, and `pytest` across a 3 OS × 2 Python matrix
  (Ubuntu/Windows/macOS × 3.11/3.12), plus a wheel-build job. All third-party
  actions are pinned by SHA.
- **Security** — `.github/workflows/security.yml` runs CodeQL on push, PR, and
  weekly cron, plus Dependency Review on pull requests.
- **Docs** — MkDocs Material site under `docs/`, deployed to GitHub Pages by
  `.github/workflows/docs.yml` on every push to `main` that touches `docs/**`
  or `mkdocs.yml`. Live at <https://linuxerlv.github.io/omniparser-fork/>.
- **Releases** — `.github/workflows/release.yml` triggers on `v*.*.*` tags,
  builds all packages with `uv build --all-packages`, and attaches the
  resulting wheels and sdists to a GitHub Release. PyPI publishing via Trusted
  Publishing is wired but gated to stable tags only.
- **Dependabot** — `.github/dependabot.yml` groups weekly `uv` and
  `github-actions` updates. Heavy ML pins (`torch`, `torchvision`,
  `ultralytics`, `transformers`) are ignored to avoid noisy churn.
- **Branch protection** — `main` requires ten status checks (`lint`,
  `mypy --strict`, six `pytest` matrix cells, `build wheels`, `codeql`),
  linear history, no force-pushes, no deletions, and resolved review
  conversations.
- **Developer tooling** — `ruff` (lint + format), `mypy --strict`, `pytest`
  with `--import-mode=importlib`, and a root `pyproject.toml` that pins all
  workspace cross-dependencies.

### Notes

- **Licensing.** The fork itself is MIT. The upstream snapshot under
  `upstream/` retains its CC-BY-4.0 license. The YOLO icon-detection weights
  carry an AGPL-3.0 obligation that downstream users must evaluate
  independently; see `docs/internal/migration-plan.md`.
- **Weights are not vendored.** `huggingface_hub.snapshot_download` fetches
  them on first use. Set `HF_ENDPOINT` to override the mirror.
- **Tests do not require network access, API keys, weights, or datasets.**
  Anything that does is marked `@pytest.mark.slow` and skipped by default.

[Unreleased]: https://github.com/linuxerlv/omniparser-fork/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/linuxerlv/omniparser-fork/releases/tag/v0.1.0a1
