"""Smoke tests for the core platform schemas.

These guarantee the contract callers depend on:
- JSON round-trip is lossless
- Required fields are required
- Common authoring mistakes are caught early (empty prompts, duplicate task_ids)
- Score bounds are enforced
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ebit_gym.core import (
    Attempt,
    AttemptCost,
    GraderSpec,
    GradingResult,
    Task,
    TaskMaterial,
    TaskMetadata,
    TaskSet,
)


def _make_task(task_id: str = "q-001") -> Task:
    return Task(
        metadata=TaskMetadata(
            task_id=task_id,
            domain="finance.quant.interview",
            difficulty="medium",
            author="alice@ebit",
            tags=["combinatorics"],
        ),
        prompt="Compute the expected number of coin flips until two heads in a row.",
        materials=[],
        grader_spec=GraderSpec(type="exact_match", config={"answer": "6"}),
        reference_answer="6",
    )


def test_task_roundtrip_json():
    t = _make_task()
    blob = t.model_dump_json()
    t2 = Task.model_validate_json(blob)
    assert t == t2


def test_taskset_roundtrip_with_splits():
    ts = TaskSet(
        name="finance-quant-v0",
        version="0.1.0",
        description="Reference quant interview questions.",
        tasks=[_make_task("q-001"), _make_task("q-002")],
        splits={"train": ["q-001"], "eval": ["q-002"]},
        provenance={"author_org": "ebit", "license": "Apache-2.0"},
    )
    blob = ts.model_dump_json()
    ts2 = TaskSet.model_validate_json(blob)
    assert ts == ts2
    assert ts2.task_by_id("q-002").metadata.task_id == "q-002"


def test_taskset_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="duplicate"):
        TaskSet(
            name="dupes",
            tasks=[_make_task("q-001"), _make_task("q-001")],
        )


def test_task_rejects_empty_prompt():
    with pytest.raises(ValidationError, match="empty"):
        Task(
            metadata=TaskMetadata(
                task_id="q-bad", domain="x", author="alice"
            ),
            prompt="   \n  ",
            grader_spec=GraderSpec(type="exact_match"),
        )


def test_task_rejects_unknown_field():
    # extra='forbid' is the cheap defense against typos in authored JSON.
    raw = {
        "metadata": {
            "task_id": "q-001",
            "domain": "x",
            "author": "a",
            "rgbic": "purple",  # typo, not a real field
        },
        "prompt": "hi",
        "grader_spec": {"type": "exact_match"},
    }
    with pytest.raises(ValidationError):
        Task.model_validate(raw)


def test_attempt_roundtrip():
    a = Attempt(
        attempt_id="att-001",
        task_id="q-001",
        task_version="0.1.0",
        model="anthropic:claude-4.6-sonnet",
        raw_response="The answer is 6.",
        parsed_answer="6",
        cost=AttemptCost(input_tokens=120, output_tokens=14, usd=0.0006),
        latency_ms=850,
    )
    blob = a.model_dump_json()
    assert Attempt.model_validate_json(blob) == a


def test_grading_result_score_bounds():
    GradingResult(
        task_id="q-001",
        attempt_id="att-001",
        score=1.0,
        passed=True,
        graded_by="exact_match",
    )
    with pytest.raises(ValidationError):
        GradingResult(
            task_id="q-001",
            attempt_id="att-001",
            score=1.5,
            passed=True,
            graded_by="exact_match",
        )
    with pytest.raises(ValidationError):
        GradingResult(
            task_id="q-001",
            attempt_id="att-001",
            score=-0.1,
            passed=False,
            graded_by="exact_match",
        )


def test_task_json_is_human_friendly():
    """The corpus authoring workflow involves humans reading and editing JSON.
    Verify the serialized form is well-shaped enough for that."""
    t = _make_task()
    parsed = json.loads(t.model_dump_json())
    assert parsed["metadata"]["task_id"] == "q-001"
    assert parsed["grader_spec"]["type"] == "exact_match"
    assert parsed["reference_answer"] == "6"


def test_task_material_kinds():
    m = TaskMaterial(kind="text", content="Some reference text.")
    assert m.kind == "text"
    m2 = TaskMaterial(
        kind="table",
        content='{"open": [1,2], "close": [2,3]}',
        mime_type="application/json",
        name="market_window.json",
    )
    assert m2.name == "market_window.json"
