"""omniparser-eval — benchmark harnesses for OmniParser.

Migrates ``upstream/eval/ss_pro_gpt4o_omniv2.py`` into a typed, testable
package with a CLI entry point ``omniparser-eval``.

Public surface:

* :class:`GroundModel` — Protocol for grounding backends
* :class:`OpenAIGroundModel` — concrete OpenAI implementation (parity with upstream)
* :class:`GroundResult` — typed return shape from any grounding call
* :class:`PredictionRecord` — JSONL row schema (binary-compatible with upstream
  ``logs_sspro_omniv2.json``)
* :func:`download_screenspot_pro` — fetch dataset via huggingface_hub (hf-mirror by default)
* :func:`load_jsonl`, :func:`iter_dataset` — dataset helpers
* :func:`compute_correctness`, :class:`Summary` — scoring
* :func:`run_eval` — single-call driver

Importing this top-level package has no model-loading or network side
effects.
"""

from __future__ import annotations

from omniparser_eval.data import (
    PredictionRecord,
    download_screenspot_pro,
    iter_dataset,
    load_jsonl,
    write_jsonl_record,
)
from omniparser_eval.driver import run_eval
from omniparser_eval.models import (
    GroundModel,
    GroundResult,
    OpenAIGroundModel,
    denormalize_coords,
    reformat_messages,
)
from omniparser_eval.scoring import (
    Summary,
    compute_correctness,
    point_in_bbox,
    summarize,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "GroundModel",
    "GroundResult",
    "OpenAIGroundModel",
    "PredictionRecord",
    "Summary",
    "__version__",
    "compute_correctness",
    "denormalize_coords",
    "download_screenspot_pro",
    "iter_dataset",
    "load_jsonl",
    "point_in_bbox",
    "reformat_messages",
    "run_eval",
    "summarize",
    "write_jsonl_record",
]
