"""Tests for omniparser_eval.data."""

from __future__ import annotations

import json
from pathlib import Path

from omniparser_eval.data import iter_dataset, load_jsonl, write_jsonl_record


def test_load_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    rows = [
        {"idx": 0, "img_path": "a.png", "bbox": [1, 2, 3, 4], "correctness": "correct"},
        {"idx": 1, "img_path": "b.png", "bbox": None, "correctness": "wrong"},
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    loaded = load_jsonl(path)
    assert loaded == rows


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"idx": 0}\n\n{"idx": 1}\n', encoding="utf-8")
    assert load_jsonl(path) == [{"idx": 0}, {"idx": 1}]


def test_write_jsonl_record_appends(tmp_path: Path) -> None:
    path = tmp_path / "out" / "log.jsonl"  # nested dir to verify mkdir
    write_jsonl_record(path, {"idx": 0, "pred": [10, 20]})
    write_jsonl_record(path, {"idx": 1, "pred": [30, 40]})
    assert load_jsonl(path) == [
        {"idx": 0, "pred": [10, 20]},
        {"idx": 1, "pred": [30, 40]},
    ]


def test_iter_dataset_yields_indexed_rows(tmp_path: Path) -> None:
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "app_a.json").write_text(
        json.dumps(
            [
                {"img_path": "./images/a/1.png", "prompt_to_evaluate": "click 1"},
                {"img_path": "./images/a/2.png", "prompt_to_evaluate": "click 2"},
            ],
        ),
        encoding="utf-8",
    )
    (ann_dir / "app_b.json").write_text(
        json.dumps([{"img_path": "./images/b/1.png", "prompt_to_evaluate": "click b"}]),
        encoding="utf-8",
    )

    rows = list(iter_dataset(tmp_path))
    assert [r["idx"] for r in rows] == [0, 1, 2]
    assert [r["task_filename"] for r in rows] == ["app_a", "app_a", "app_b"]


def test_iter_dataset_handles_flat_layout(tmp_path: Path) -> None:
    """When there's no annotations/ subdir, iter from root."""
    (tmp_path / "app_a.json").write_text(
        json.dumps([{"img_path": "./x.png", "prompt_to_evaluate": "click"}]),
        encoding="utf-8",
    )
    rows = list(iter_dataset(tmp_path))
    assert len(rows) == 1
    assert rows[0]["idx"] == 0
    assert rows[0]["task_filename"] == "app_a"


def test_upstream_log_schema_round_trip() -> None:
    """Parity check: load one row from upstream log, verify schema fields are preserved.

    This test reads ``upstream/eval/logs_sspro_omniv2.json`` directly to
    catch schema drift. It's marked as a contract test rather than a
    full eval; only the first line is read.
    """
    upstream_log = Path(__file__).parents[4] / "upstream" / "eval" / "logs_sspro_omniv2.json"
    if not upstream_log.exists():
        # The submodule may not be checked out in some CI contexts; skip
        # rather than fail.
        import pytest

        pytest.skip("upstream submodule not checked out")

    with upstream_log.open(encoding="utf-8") as fh:
        first = fh.readline()
    row = json.loads(first)

    expected_keys = {
        "img_path",
        "group",
        "platform",
        "application",
        "lang",
        "instruction_style",
        "prompt_to_evaluate",
        "gt_type",
        "ui_type",
        "task_filename",
        "pred",
        "raw_response",
        "bbox",
        "correctness",
        "idx",
    }
    assert expected_keys.issubset(row.keys())
    # Coordinates must be absolute pixel ints (or None).
    if row["pred"] is not None:
        assert all(isinstance(v, int) for v in row["pred"])
    if row["bbox"] is not None:
        assert len(row["bbox"]) == 4
        assert all(isinstance(v, int) for v in row["bbox"])
    # raw_response is dict or string
    assert isinstance(row["raw_response"], (dict, str))
