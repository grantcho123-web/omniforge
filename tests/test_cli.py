"""Tests for the eval CLI.

Invoke main() directly with argv lists — faster and easier to assert on
than subprocess. The model adapter under test is MockAdapter so no
network is involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniforge.cli import main
from omniforge.core import GraderSpec, Task, TaskMetadata, TaskSet


def _write_taskset(path: Path) -> TaskSet:
    ts = TaskSet(
        name="cli-test-corpus",
        version="0.1.0",
        description="Tiny set for CLI tests.",
        tasks=[
            Task(
                metadata=TaskMetadata(
                    task_id="q-1",
                    domain="test.exact",
                    author="test",
                ),
                prompt="What is 6 times 7?",
                grader_spec=GraderSpec(type="exact_match", config={"answer": "42"}),
                reference_answer="42",
            ),
            Task(
                metadata=TaskMetadata(
                    task_id="q-2",
                    domain="test.regex",
                    author="test",
                ),
                prompt="Mention London.",
                grader_spec=GraderSpec(type="regex", config={"pattern": r"London"}),
            ),
            Task(
                metadata=TaskMetadata(
                    task_id="q-3",
                    domain="test.llm_judge",
                    author="test",
                ),
                prompt="Open-ended.",
                grader_spec=GraderSpec(type="llm_judge", config={"rubric": "be good"}),
                rubric="be good",
            ),
        ],
        splits={"eval": ["q-1", "q-2"]},
    )
    path.write_text(ts.model_dump_json(indent=2))
    return ts


def test_eval_end_to_end_with_mock(tmp_path, capsys):
    """Full eval: load → run → grade → summary table."""
    ts_path = tmp_path / "set.json"
    _write_taskset(ts_path)

    # The mock:default factory ignores its responses, so we patch in a real
    # MockAdapter by re-registering a custom factory? Simpler: use the
    # default which returns "ok" for both — q-1 fails (ok != 42), q-2 fails
    # (no "London"), q-3 skipped. That's a complete coverage of the three
    # branches we care about.
    exit_code = main(
        ["eval", "--task-set", str(ts_path), "--model", "mock:default", "--ids", "q-1", "q-2"]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "cli-test-corpus" in out
    assert "Model: mock:default" in out
    assert "q-1" in out
    assert "q-2" in out
    assert "Aggregate:" in out


def test_eval_skips_llm_judge_with_warning(tmp_path, capsys):
    ts_path = tmp_path / "set.json"
    _write_taskset(ts_path)

    main(["eval", "--task-set", str(ts_path), "--model", "mock:default"])
    out = capsys.readouterr().out
    assert "[skipped]" in out
    assert "q-3" in out
    assert "llm_judge" in out


def test_eval_writes_output_json(tmp_path):
    ts_path = tmp_path / "set.json"
    out_path = tmp_path / "runs" / "result.json"
    _write_taskset(ts_path)

    main(
        [
            "eval",
            "--task-set",
            str(ts_path),
            "--model",
            "mock:default",
            "--output",
            str(out_path),
            "--ids",
            "q-1",
        ]
    )
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["taskset"]["name"] == "cli-test-corpus"
    assert payload["results"][0]["task_id"] == "q-1"
    assert payload["results"][0]["attempt"]["model"] == "mock:default"
    assert payload["results"][0]["grading"] is not None


def test_eval_filters_by_split(tmp_path, capsys):
    ts_path = tmp_path / "set.json"
    _write_taskset(ts_path)

    main(["eval", "--task-set", str(ts_path), "--model", "mock:default", "--split", "eval"])
    out = capsys.readouterr().out
    assert "q-1" in out
    assert "q-2" in out
    # q-3 not in eval split, should NOT appear in output table
    assert "q-3" not in out.split("Aggregate:")[0]


def test_eval_rejects_unknown_model(tmp_path, capsys):
    ts_path = tmp_path / "set.json"
    _write_taskset(ts_path)

    exit_code = main(
        ["eval", "--task-set", str(ts_path), "--model", "nonexistent:model"]
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "unknown model" in err


def test_list_models(capsys):
    main(["list-models"])
    out = capsys.readouterr().out
    assert "anthropic:claude-4.6-sonnet" in out
    assert "openai:gpt-4o-mini" in out
    assert "upstage:solar-pro" in out


def test_list_graders(capsys):
    main(["list-graders"])
    out = capsys.readouterr().out
    assert "exact_match" in out
    assert "regex" in out
    assert "composite" in out
    assert "human" in out
    assert "llm_judge" in out  # listed with caveat
    assert "programmatic only" in out


def test_inspect_taskset(tmp_path, capsys):
    ts_path = tmp_path / "set.json"
    _write_taskset(ts_path)
    main(["inspect-taskset", str(ts_path)])
    out = capsys.readouterr().out
    assert "cli-test-corpus" in out
    assert "tasks:       3" in out
    assert "test.exact: 1" in out
    assert "exact_match: 1" in out
    assert "regex: 1" in out


def test_eval_with_export_writes_jsonl(tmp_path):
    """The --export flag should produce a lab-consumable JSONL alongside the run."""
    ts_path = tmp_path / "set.json"
    out_path = tmp_path / "run" / "results.json"
    export_path = tmp_path / "run" / "export.jsonl"
    _write_taskset(ts_path)

    main(
        [
            "eval",
            "--task-set",
            str(ts_path),
            "--model",
            "mock:default",
            "--output",
            str(out_path),
            "--export",
            "generic",
            "--export-include-failed",  # mock fails everything; need this for non-empty file
        ]
    )
    assert export_path.exists()
    lines = [line for line in export_path.read_text().splitlines() if line]
    # mock:default only ships one response; the second task gets an
    # "exhausted" error and is correctly dropped by the exporter.
    # q-3 (llm_judge) is skipped at grading time. So exactly one record lands.
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["task_id"] == "q-1"
    assert payload["completion"] == "ok"


def test_no_subcommand_errors():
    with pytest.raises(SystemExit):
        main([])
