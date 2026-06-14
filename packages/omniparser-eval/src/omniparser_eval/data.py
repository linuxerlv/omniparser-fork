"""Dataset loading and JSONL I/O for ScreenSpot-Pro and other benchmarks.

JSONL schema is binary-compatible with
``upstream/eval/logs_sspro_omniv2.json``. See :class:`PredictionRecord`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "PredictionRecord",
    "download_screenspot_pro",
    "iter_dataset",
    "load_jsonl",
    "write_jsonl_record",
]

_logger = logging.getLogger(__name__)

#: Default Hugging Face mirror for users in mainland China. Override via
#: the ``HF_ENDPOINT`` environment variable or the ``mirror=`` argument.
HF_MIRROR_DEFAULT = "https://hf-mirror.com"

#: Hugging Face dataset repo for ScreenSpot-Pro.
SCREENSPOT_PRO_REPO = "likaixin/ScreenSpot-Pro"


class PredictionRecord(TypedDict, total=False):
    """One JSONL row.

    Schema is byte-for-byte identical to ``upstream/eval/logs_sspro_omniv2.json``.
    All coordinate fields are absolute pixels in the source image.

    Required fields (always present):

    * ``img_path``: relative path under the dataset root
    * ``group``, ``platform``, ``application``, ``lang``, ``instruction_style``,
      ``prompt_to_evaluate``, ``gt_type``, ``ui_type``, ``task_filename``: pass-through
      from the dataset row
    * ``bbox``: ``[x1, y1, x2, y2]`` GT bbox in absolute pixels
    * ``idx``: integer row index

    Filled in by the eval driver:

    * ``pred``: ``[x, y]`` predicted click point in absolute pixels, or ``None``
    * ``raw_response``: dict (parsed JSON) or string from the model
    * ``correctness``: scoring verdict (see :mod:`omniparser_eval.scoring`)
    """

    img_path: str
    group: str
    platform: str
    application: str
    lang: str
    instruction_style: str
    prompt_to_evaluate: str
    gt_type: Literal["positive", "negative"]
    ui_type: str
    task_filename: str
    bbox: list[int] | None
    idx: int
    pred: list[int] | None
    raw_response: dict[str, Any] | str | None
    correctness: Literal[
        "correct",
        "wrong",
        "negative_correct",
        "negative_wrong",
        "failed",
    ]


def download_screenspot_pro(
    *,
    cache_dir: Path | str | None = None,
    mirror: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download ScreenSpot-Pro via huggingface_hub and return local path.

    Args:
        cache_dir: where to cache the dataset. Default: ``$HF_HOME`` or
            ``~/.cache/huggingface``.
        mirror: HF endpoint URL. Default: ``HF_ENDPOINT`` env var if set,
            otherwise :data:`HF_MIRROR_DEFAULT` (hf-mirror.com). Pass
            ``"hf"`` to force the official endpoint.
        revision: optional git revision (commit/tag/branch).

    Returns:
        Path to the local snapshot directory.
    """
    from huggingface_hub import snapshot_download

    if mirror is None:
        mirror = os.environ.get("HF_ENDPOINT", HF_MIRROR_DEFAULT)
    if mirror == "hf":
        mirror = "https://huggingface.co"
    elif mirror == "hf-mirror":
        mirror = HF_MIRROR_DEFAULT

    # huggingface_hub respects HF_ENDPOINT; set it for the duration of the call.
    prev = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = mirror
    try:
        local_path = snapshot_download(
            repo_id=SCREENSPOT_PRO_REPO,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
            revision=revision,
        )
    finally:
        if prev is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = prev

    _logger.info("ScreenSpot-Pro downloaded to %s (mirror=%s)", local_path, mirror)
    return Path(local_path)


def load_jsonl(path: Path | str) -> list[PredictionRecord]:
    """Load a JSONL file into a list of records."""
    records: list[PredictionRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def iter_dataset(dataset_root: Path | str) -> Iterator[PredictionRecord]:
    """Yield rows from the ScreenSpot-Pro annotations.

    The dataset on Hugging Face stores per-application JSON files under
    ``annotations/`` plus a flat ``images/`` tree. We yield rows in the
    same order upstream's eval log uses (sorted by application, then by
    file index) so that ``idx`` values line up with upstream when run on
    the same dataset snapshot.
    """
    dataset_root = Path(dataset_root)
    annotations_dir = dataset_root / "annotations"
    if not annotations_dir.exists():
        # Some snapshots place annotations at the root.
        annotations_dir = dataset_root

    idx = 0
    for ann_file in sorted(annotations_dir.glob("*.json")):
        with ann_file.open(encoding="utf-8") as fh:
            ann_data = json.load(fh)
        if not isinstance(ann_data, list):
            continue
        for row in ann_data:
            row["idx"] = idx
            row["task_filename"] = ann_file.stem
            idx += 1
            yield row


def write_jsonl_record(path: Path | str, record: PredictionRecord) -> None:
    """Append a single record to a JSONL file (creating it if needed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
