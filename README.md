# OmniParser Monorepo

> Engineering-grade rework of [microsoft/OmniParser](https://github.com/microsoft/OmniParser) (upstream commit `b0d5c9f5`, v.2.0.1) into a standards-conformant Python monorepo.

This repository is a clean-room engineering reorganization of OmniParser. The upstream project is a high-impact research artifact but ships without packaging, tests, CI, type checking, or modular boundaries. This monorepo consumes the upstream source under MIT terms (see `LICENSE` and `NOTICE`) and reorganizes it into installable, independently versioned packages with industry-standard tooling.

## Status

Work in progress. The directory skeleton is in place; package contents are being migrated from `upstream/` and refactored. See `docs/migration-plan.md` (forthcoming) for the staged plan.

## Layout

```
ominpaser/
|-- packages/
|   |-- omniparser-core/        # Pure inference library (YOLO + caption + OCR + IoU dedup)
|   |-- omniparser-server/      # FastAPI HTTP service (depends on omniparser-core)
|   `-- omniparser-eval/        # ScreenSpot-Pro and other benchmark harnesses
|-- apps/
|   |-- gradio-demo/            # Single-page Gradio demo
|   `-- computer-use-agent/     # VLM agent loop (consolidated from upstream omnitool/gradio/)
|-- examples/                   # Runnable usage examples (replaces demo.ipynb)
|-- benchmarks/                 # Reproducible benchmark configs and reports
|-- docs/                       # mkdocs-material site source
|-- scripts/                    # Cross-package automation
|-- upstream/                   # Pinned reference clone (read-only, not part of the build)
`-- .github/workflows/          # CI matrix, release, security scans
```

## Upstream pin

`upstream/` is a `git clone` of microsoft/OmniParser at commit `b0d5c9f5701f7e2be4771872e6e928da77759df3`. It is kept as an **immutable reference** for traceability and is excluded from the workspace build. All migrated code is rewritten under each `packages/*/src/` tree.

## License

This monorepo is licensed under the MIT License (see `LICENSE`). The upstream OmniParser repository is released under CC-BY-4.0; the rewritten code in `packages/` and `apps/` is original or derivative work that we are relicensing as MIT, with attribution preserved in `NOTICE`.

Model weights distributed by Microsoft on Hugging Face (`microsoft/OmniParser-v2.0`) carry their own licenses (notably AGPL-3.0 for the `icon_detect` YOLO weights). Weights are downloaded at runtime via `huggingface_hub` and are **not** redistributed in this repository. Downstream consumers are responsible for compliance with the weights' licenses.

## Engineering standards

Documented in `docs/engineering-standards.md` (forthcoming). Headline choices:

- **Build system**: uv workspaces, PEP 621 `pyproject.toml`, src layout
- **Lint + format**: `ruff` (single tool replaces black + isort + flake8)
- **Type checking**: `mypy --strict` on `packages/`, relaxed on `apps/`
- **Tests**: `pytest` + `pytest-cov`, smoke tests gate every PR
- **CI**: GitHub Actions matrix (py3.10/3.11/3.12, Linux/Windows), trusted-publish to PyPI on tag
- **Pre-commit**: `ruff`, `ruff format`, `mypy`, `gitleaks`, basic file hygiene
- **Release**: Conventional Commits + `release-please`, SemVer per package
- **Supply chain**: `pip-audit`, Dependabot, SBOM via CycloneDX on release
