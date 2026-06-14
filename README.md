# OmniParser Monorepo

[![ci](https://github.com/linuxerlv/omniparser-fork/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/linuxerlv/omniparser-fork/actions/workflows/ci.yml)
[![security](https://github.com/linuxerlv/omniparser-fork/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/linuxerlv/omniparser-fork/actions/workflows/security.yml)
[![docs](https://github.com/linuxerlv/omniparser-fork/actions/workflows/docs.yml/badge.svg?branch=main)](https://linuxerlv.github.io/omniparser-fork/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/typecheck-mypy%20--strict-2A6DB2)](https://mypy-lang.org/)

> Engineering-grade rework of [microsoft/OmniParser](https://github.com/microsoft/OmniParser) (upstream commit [`b0d5c9f5`](https://github.com/microsoft/OmniParser/commit/b0d5c9f5701f7e2be4771872e6e928da77759df3), v2.0.1) into a standards-conformant Python `uv` workspace monorepo.

The upstream project is a high-impact research artifact but ships without packaging, tests, CI, type checking, or modular boundaries. This monorepo consumes the upstream source under MIT terms (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)) and reorganizes it into installable, independently versioned packages with industry-standard tooling. Documentation site: **<https://linuxerlv.github.io/omniparser-fork/>**.

## Status

**Active development.** The full monorepo skeleton, refactored core/server/client/eval packages, CI/CD pipelines, and documentation site are in place. The upstream `b0d5c9f5` reference is pinned via git submodule under `upstream/` and excluded from the build. See [`docs/internal/migration-plan.md`](docs/internal/migration-plan.md) for the staged plan.

| Surface                       | State                                                            |
| ----------------------------- | ---------------------------------------------------------------- |
| Workspace + lockfile          | ✅ `uv` workspaces, 7 members, single `uv.lock`                  |
| Lint + format                 | ✅ `ruff check` + `ruff format` (CI gates)                       |
| Type checking                 | ✅ `mypy --strict` over all `packages/*/src` trees               |
| Tests                         | ✅ `pytest` 132 tests, 3 OS × 2 py matrix on every push          |
| Build                         | ✅ `uv build --all-packages` produces 7 wheels                   |
| Docs                          | ✅ MkDocs Material site auto-deployed to GitHub Pages            |
| Security                      | ✅ CodeQL (Python) + dependency-review on PRs                    |
| Dependency updates            | ✅ Dependabot weekly grouped (`uv` + `github-actions`)           |
| Release automation            | ✅ Tag `v*.*.*` → GitHub Release with built wheels (first tag: [`v0.1.0a1`](https://github.com/linuxerlv/omniparser-fork/releases/tag/v0.1.0a1)) |
| Branch protection             | ✅ `main` requires 10 status checks, linear history, no force-pushes |
| PyPI publishing               | ⏳ Trusted-publishing wired; gated to stable tags (skipped for pre-releases) |

## Packages

| Path                                                       | Name                | Purpose                                                                          |
| ---------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------- |
| [`packages/omniparser-core`](packages/omniparser-core)     | `omniparser-core`   | Pluggable detector + caption + OCR backends with IoU-dedup pipeline              |
| [`packages/omniparser-server`](packages/omniparser-server) | `omniparser-server` | FastAPI inference service wrapping `omniparser-core`                             |
| [`packages/omniparser-client`](packages/omniparser-client) | `omniparser-client` | Lightweight HTTP client for the inference server (no `torch` dependency)         |
| [`packages/omniparser-eval`](packages/omniparser-eval)     | `omniparser-eval`   | Benchmark harnesses (ScreenSpot-Pro, GPT-4X grounding) with verbatim prompts     |
| [`packages/omnitool-agent`](packages/omnitool-agent)       | `omnitool-agent`    | Adapter bridging `omniparser-client` output into the legacy omnitool dict shape  |
| [`apps/parser-demo`](apps/parser-demo)                     | `parser-demo`       | Single-page Gradio demo for `omniparser-core`                                    |
| [`apps/omnitool-ui`](apps/omnitool-ui)                     | `omnitool-ui`       | Gradio agent UI driving `omnitool-agent`                                         |

## Quick start

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
git clone https://github.com/linuxerlv/omniparser-fork.git
cd omniparser-fork
git submodule update --init                   # pull pinned upstream/ reference

uv sync --all-packages --group dev            # install workspace + dev tools
uv run --no-sync pytest -q                    # run the test suite
uv run --no-sync ruff check                   # lint
uv run --no-sync mypy packages/*/src          # type-check
uv build --all-packages --out-dir dist        # build all wheels
```

The full getting-started walkthrough lives in [`docs/getting-started.md`](docs/getting-started.md). Architecture and package boundaries: [`docs/architecture.md`](docs/architecture.md).

## Upstream pin

`upstream/` is a git submodule pinned at upstream commit [`b0d5c9f5`](https://github.com/microsoft/OmniParser/commit/b0d5c9f5701f7e2be4771872e6e928da77759df3). It is treated as an **immutable reference** for traceability and is excluded from the workspace build (`tool.uv.workspace.exclude`). All migrated code is rewritten under each package's `src/` tree.

## License

This monorepo is licensed under the [MIT License](LICENSE). The upstream OmniParser repository is released under CC-BY-4.0; the rewritten code in `packages/` and `apps/` is original or derivative work relicensed as MIT, with attribution preserved in [`NOTICE`](NOTICE).

Model weights distributed by Microsoft on Hugging Face (`microsoft/OmniParser-v2.0`) carry their own licenses — notably **AGPL-3.0** for the `icon_detect` YOLO weights. Weights are downloaded at runtime via `huggingface_hub` (default mirror: `https://hf-mirror.com`) and are **not** redistributed in this repository. Downstream consumers are responsible for compliance with the weights' licenses.

## Engineering standards

- **Build**: [uv](https://docs.astral.sh/uv/) workspaces, [PEP 621](https://peps.python.org/pep-0621/) + [PEP 735](https://peps.python.org/pep-0735/) `pyproject.toml`, src layout, [Hatchling](https://hatch.pypa.io/) backend.
- **Python**: `>=3.11,<3.13`. Tested on CPython 3.11 and 3.12.
- **Lint + format**: [`ruff`](https://docs.astral.sh/ruff/) (replaces black + isort + flake8 + pyupgrade).
- **Type checking**: [`mypy --strict`](https://mypy.readthedocs.io/) on every `packages/*/src` tree.
- **Tests**: [`pytest`](https://docs.pytest.org/) with `--import-mode=importlib`, no network and no model weights in the default lane (`@pytest.mark.slow` for opt-in).
- **CI**: GitHub Actions, all action versions pinned by SHA. Matrix is 3 OS × 2 py for `pytest`.
- **Security**: CodeQL (Python) on push/PR/weekly + GitHub Dependency Review on PRs.
- **Release**: tag `v*.*.*` triggers `release.yml` → `uv build --all-packages` → upload artifacts → GitHub Release.
- **Docs**: MkDocs Material + [`mkdocstrings[python]`](https://mkdocstrings.github.io/) auto-deployed to GitHub Pages on every push to `main` that touches `docs/**`.
