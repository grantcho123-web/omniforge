"""Tests for the export adapters."""
from __future__ import annotations

import json
from pathlib import Path

from omniforge.core.export import (
    export_anthropic_jsonl,
    export_generic_jsonl,
    export_openai_finetune_jsonl,
)
from omniforge.core.task import (
    Attempt,
    GraderSpec,
    GradingResult,
    Task,
    TaskMaterial,
    TaskMetadata,
)


def _triple(
    task_id: str,
    *,
    passed: bool = True,
    score: float = 1.0,
    response: str = "42",
    has_grade: bool = True,
    error: str | None = None,
    materials: list[TaskMaterial] | None = None,
) -> tuple[Task, Attempt, GradingResult | None]:
    task = Task(
        metadata=TaskMetadata(task_id=task_id, domain="x", author="a"),
        prompt="What is 6 times 7?",
        materials=materials or [],
        grader_spec=GraderSpec(type="exact_match", config={"answer": "42"}),
        reference_answer="42",
    )
    attempt = Attempt(
        attempt_id=f"att-{task_id}",
        task_id=task_id,
        task_version="0.1.0",
        model="anthropic:claude-sonnet-4-5",
        raw_response=response,
        error=error,
    )
    grade: GradingResult | None = None
    if has_grade:
        grade = GradingResult(
            task_id=task_id,
            attempt_id=f"att-{task_id}",
            score=score,
            passed=passed,
            graded_by="exact_match",
        )
    return task, attempt, grade


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ------------------------------------------------------------- openai


def test_openai_finetune_basic(tmp_path):
    out = tmp_path / "ft.jsonl"
    n = export_openai_finetune_jsonl(
        [_triple("q-1"), _triple("q-2", response="forty-two")],
        out,
    )
    assert n == 2
    lines = _read_jsonl(out)
    assert lines[0]["messages"][0]["role"] == "user"
    assert lines[0]["messages"][0]["content"] == "What is 6 times 7?"
    assert lines[0]["messages"][1]["role"] == "assistant"
    assert lines[0]["messages"][1]["content"] == "42"


def test_openai_finetune_with_system_prompt(tmp_path):
    out = tmp_path / "ft.jsonl"
    export_openai_finetune_jsonl(
        [_triple("q-1")],
        out,
        system_prompt="You are a careful quant.",
    )
    lines = _read_jsonl(out)
    assert lines[0]["messages"][0]["role"] == "system"
    assert lines[0]["messages"][0]["content"] == "You are a careful quant."
    assert lines[0]["messages"][1]["role"] == "user"


def test_openai_finetune_excludes_failed_by_default(tmp_path):
    out = tmp_path / "ft.jsonl"
    n = export_openai_finetune_jsonl(
        [_triple("q-pass", passed=True), _triple("q-fail", passed=False, score=0.2)],
        out,
    )
    assert n == 1
    lines = _read_jsonl(out)
    # q-fail filtered out
    assistant_contents = [m for line in lines for m in line["messages"] if m["role"] == "assistant"]
    assert all("forty" not in c["content"] for c in assistant_contents)


def test_openai_finetune_can_include_failed(tmp_path):
    out = tmp_path / "ft.jsonl"
    n = export_openai_finetune_jsonl(
        [_triple("q-pass", passed=True), _triple("q-fail", passed=False, score=0.2)],
        out,
        include_failed=True,
    )
    assert n == 2


def test_export_skips_errored_attempts(tmp_path):
    out = tmp_path / "ft.jsonl"
    n = export_openai_finetune_jsonl(
        [_triple("q-1"), _triple("q-2", error="network down")],
        out,
        include_failed=True,
    )
    # The errored attempt is dropped regardless of include_failed
    assert n == 1


def test_export_skips_ungraded_attempts(tmp_path):
    out = tmp_path / "ft.jsonl"
    n = export_openai_finetune_jsonl(
        [_triple("q-1"), _triple("q-2", has_grade=False)],
        out,
    )
    assert n == 1


def test_export_renders_materials_into_user_message(tmp_path):
    out = tmp_path / "ft.jsonl"
    materials = [TaskMaterial(kind="text", content="Aux info.", name="aux.txt")]
    export_openai_finetune_jsonl([_triple("q-1", materials=materials)], out)
    lines = _read_jsonl(out)
    user_content = lines[0]["messages"][0]["content"]
    assert "What is 6 times 7?" in user_content
    assert "--- material 1 (aux.txt) ---" in user_content
    assert "Aux info." in user_content


# ----------------------------------------------------------- anthropic


def test_anthropic_basic(tmp_path):
    out = tmp_path / "a.jsonl"
    export_anthropic_jsonl([_triple("q-1")], out)
    [line] = _read_jsonl(out)
    assert "system" not in line  # absent when not provided
    assert line["messages"][0]["role"] == "user"
    assert line["messages"][1]["role"] == "assistant"
    assert line["messages"][1]["content"] == "42"


def test_anthropic_with_system_prompt(tmp_path):
    out = tmp_path / "a.jsonl"
    export_anthropic_jsonl([_triple("q-1")], out, system_prompt="be terse")
    [line] = _read_jsonl(out)
    assert line["system"] == "be terse"


# ----------------------------------------------------------- generic


def test_generic_basic(tmp_path):
    out = tmp_path / "g.jsonl"
    n = export_generic_jsonl(
        [_triple("q-1", score=1.0), _triple("q-2", passed=False, score=0.3)],
        out,
    )
    # generic includes failed by default
    assert n == 2
    lines = _read_jsonl(out)
    assert lines[0]["task_id"] == "q-1"
    assert lines[0]["prompt"] == "What is 6 times 7?"
    assert lines[0]["completion"] == "42"
    assert lines[0]["score"] == 1.0
    assert lines[0]["passed"] is True
    assert lines[0]["model"] == "anthropic:claude-sonnet-4-5"
    assert lines[1]["passed"] is False
    assert lines[1]["score"] == 0.3


def test_generic_can_filter_to_passed_only(tmp_path):
    out = tmp_path / "g.jsonl"
    n = export_generic_jsonl(
        [_triple("q-1"), _triple("q-2", passed=False, score=0.2)],
        out,
        include_failed=False,
    )
    assert n == 1


def test_export_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deep" / "data.jsonl"
    n = export_openai_finetune_jsonl([_triple("q-1")], out)
    assert n == 1
    assert out.exists()


def test_export_writes_utf8_for_non_ascii(tmp_path):
    """Korean content must survive the round trip unmangled."""
    out = tmp_path / "ko.jsonl"
    triple = _triple("q-ko", response="네, 좋습니다.")
    export_openai_finetune_jsonl([triple], out)
    line = _read_jsonl(out)[0]
    assert line["messages"][1]["content"] == "네, 좋습니다."
