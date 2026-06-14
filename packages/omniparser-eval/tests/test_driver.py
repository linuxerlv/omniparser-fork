"""Tests for omniparser_eval.driver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from omniparser_eval.data import load_jsonl
from omniparser_eval.driver import run_eval
from omniparser_eval.models import GroundResult


class _StubModel:
    """Stub GroundModel that returns a deterministic prediction per call."""

    def __init__(self, *, point: list[int] | None = None, result: str = "positive") -> None:
        self.point = point
        self.result = result
        self.calls: list[tuple[str, Any]] = []

    def _make_result(self) -> GroundResult:
        return GroundResult(
            result=self.result,  # type: ignore[typeddict-item]
            bbox=None,
            point=self.point,
            raw_response={"stub": True},
        )

    def ground_only_positive(self, instruction: str, image: Image.Image) -> GroundResult:
        self.calls.append((instruction, image.size))
        return self._make_result()

    def ground_allow_negative(self, instruction: str, image: Image.Image) -> GroundResult:
        self.calls.append((instruction, image.size))
        return self._make_result()

    def ground_with_uncertainty(self, instruction: str, image: Image.Image) -> GroundResult:
        self.calls.append((instruction, image.size))
        return self._make_result()


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """Build a minimal local dataset with 2 rows + matching PNG files."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Two images, two annotations.
    Image.new("RGB", (200, 100), color=(255, 0, 0)).save(images_dir / "a.png")
    Image.new("RGB", (200, 100), color=(0, 255, 0)).save(images_dir / "b.png")

    ann = [
        {
            "img_path": "./images/a.png",
            "group": "G1",
            "platform": "linux",
            "application": "app",
            "lang": "en",
            "instruction_style": "instruction",
            "prompt_to_evaluate": "click red",
            "gt_type": "positive",
            "ui_type": "icon",
            "bbox": [50, 25, 150, 75],
        },
        {
            "img_path": "./images/b.png",
            "group": "G1",
            "platform": "linux",
            "application": "app",
            "lang": "en",
            "instruction_style": "instruction",
            "prompt_to_evaluate": "click green",
            "gt_type": "positive",
            "ui_type": "icon",
            "bbox": [10, 10, 30, 30],
        },
    ]
    (ann_dir / "tasks.json").write_text(json.dumps(ann), encoding="utf-8")
    return tmp_path


def test_run_eval_writes_jsonl_and_scores_correct(mini_dataset: Path) -> None:
    out = mini_dataset / "out.jsonl"
    model = _StubModel(point=[100, 50])  # inside row 0 bbox; outside row 1 bbox
    summary = run_eval(
        model=model,
        dataset_root=mini_dataset,
        output_path=out,
        method="positive",
    )
    assert summary.total == 2
    assert summary.correct == 1
    assert summary.wrong == 1
    rows = load_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["correctness"] == "correct"
    assert rows[1]["correctness"] == "wrong"
    assert rows[0]["pred"] == [100, 50]
    assert rows[0]["raw_response"] == {"stub": True}


def test_run_eval_max_rows(mini_dataset: Path) -> None:
    out = mini_dataset / "out.jsonl"
    model = _StubModel(point=[100, 50])
    summary = run_eval(
        model=model,
        dataset_root=mini_dataset,
        output_path=out,
        method="positive",
        max_rows=1,
    )
    assert summary.total == 1
    assert len(model.calls) == 1


def test_run_eval_resume_skips_completed(mini_dataset: Path) -> None:
    out = mini_dataset / "out.jsonl"
    # Pre-populate output with idx=0 already done.
    out.write_text(
        json.dumps({"idx": 0, "img_path": "./images/a.png", "correctness": "correct"}) + "\n",
        encoding="utf-8",
    )
    model = _StubModel(point=[100, 50])
    summary = run_eval(
        model=model,
        dataset_root=mini_dataset,
        output_path=out,
        method="positive",
    )
    # Only idx=1 should have been called.
    assert len(model.calls) == 1
    assert summary.total == 1


def test_run_eval_no_resume_processes_all(mini_dataset: Path) -> None:
    out = mini_dataset / "out.jsonl"
    out.write_text(
        json.dumps({"idx": 0, "img_path": "./images/a.png", "correctness": "correct"}) + "\n",
        encoding="utf-8",
    )
    model = _StubModel(point=[100, 50])
    summary = run_eval(
        model=model,
        dataset_root=mini_dataset,
        output_path=out,
        method="positive",
        resume=False,
    )
    assert len(model.calls) == 2
    assert summary.total == 2


def test_run_eval_handles_missing_image(tmp_path: Path) -> None:
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "img_path": "./images/missing.png",
                    "prompt_to_evaluate": "x",
                    "gt_type": "positive",
                    "bbox": [0, 0, 10, 10],
                },
            ],
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    model = _StubModel(point=[5, 5])
    summary = run_eval(
        model=model,
        dataset_root=tmp_path,
        output_path=out,
        method="positive",
    )
    # Image load failed, model not called, verdict=failed.
    assert len(model.calls) == 0
    assert summary.failed == 1
    rows = load_jsonl(out)
    assert rows[0]["correctness"] == "failed"
    assert rows[0]["pred"] is None
