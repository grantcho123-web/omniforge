"""Tests for the attempt runner."""
from __future__ import annotations

import pytest

from omniforge.core import GraderSpec, Task, TaskMaterial, TaskMetadata, TaskSet
from omniforge.core.runner import AttemptRunner, RunnerConfig
from omniforge.models.mock import MockAdapter


def _task(task_id: str = "t-1", materials: list[TaskMaterial] | None = None) -> Task:
    return Task(
        metadata=TaskMetadata(
            task_id=task_id, version="0.1.0", domain="finance.quant", author="alice"
        ),
        prompt="What is 6 times 7?",
        materials=materials or [],
        grader_spec=GraderSpec(type="exact_match", config={"answer": "42"}),
        reference_answer="42",
    )


def test_run_single_task_basic():
    model = MockAdapter(responses=["The answer is 42."], name="mock:test")
    runner = AttemptRunner(model)
    attempt = runner.run(_task())

    assert attempt.task_id == "t-1"
    assert attempt.task_version == "0.1.0"
    assert attempt.model == "mock:test"
    assert attempt.raw_response == "The answer is 42."
    assert attempt.error is None
    assert attempt.attempt_id  # UUID present
    assert attempt.started_at <= attempt.completed_at


def test_run_attaches_cost_and_latency():
    model = MockAdapter(responses=["hi"], latency_ms=42)
    runner = AttemptRunner(model)
    attempt = runner.run(_task())

    assert attempt.cost is not None
    assert attempt.cost.input_tokens is not None
    assert attempt.cost.output_tokens is not None
    assert attempt.cost.usd == 0.0
    assert attempt.latency_ms == 42


def test_run_records_error_attempts_not_raise():
    model = MockAdapter(responses=["x"])
    model.call("burn it")  # exhaust the response
    runner = AttemptRunner(model)
    attempt = runner.run(_task())
    # Adapter exhausted → error surfaced, not raised
    assert attempt.error == "MockAdapter exhausted"
    assert attempt.raw_response == ""


def test_run_passes_runner_config_to_model():
    model = MockAdapter(responses=["ok"])
    runner = AttemptRunner(
        model,
        RunnerConfig(system_prompt="be brief", max_tokens=64, temperature=0.5),
    )
    runner.run(_task())
    assert model.calls[0]["system"] == "be brief"
    assert model.calls[0]["max_tokens"] == 64
    assert model.calls[0]["temperature"] == 0.5


def test_run_renders_materials_into_prompt():
    materials = [
        TaskMaterial(kind="text", content="Reference text A.", name="a.txt"),
        TaskMaterial(kind="code", content="def f(): return 1"),
    ]
    model = MockAdapter(responses=["ok"])
    runner = AttemptRunner(model)
    runner.run(_task(materials=materials))

    rendered = model.calls[0]["prompt"]
    assert "What is 6 times 7?" in rendered
    assert "--- material 1 (a.txt) ---" in rendered
    assert "Reference text A." in rendered
    assert "--- material 2 ---" in rendered
    assert "def f(): return 1" in rendered


def test_run_no_materials_passes_prompt_unchanged():
    model = MockAdapter(responses=["ok"])
    runner = AttemptRunner(model)
    runner.run(_task())
    assert model.calls[0]["prompt"] == "What is 6 times 7?"


# ----------------------------------------------------------------- run_set


def _taskset() -> TaskSet:
    return TaskSet(
        name="mini",
        tasks=[_task("a"), _task("b"), _task("c")],
        splits={"train": ["a", "b"], "eval": ["c"]},
    )


def test_run_set_all_tasks_by_default():
    model = MockAdapter(responses=["1", "2", "3"])
    runner = AttemptRunner(model)
    attempts = runner.run_set(_taskset())
    assert [a.task_id for a in attempts] == ["a", "b", "c"]
    assert [a.raw_response for a in attempts] == ["1", "2", "3"]


def test_run_set_filters_by_split():
    model = MockAdapter(responses=["e"])
    runner = AttemptRunner(model)
    attempts = runner.run_set(_taskset(), split="eval")
    assert [a.task_id for a in attempts] == ["c"]


def test_run_set_filters_by_ids():
    model = MockAdapter(responses=["x", "y"])
    runner = AttemptRunner(model)
    attempts = runner.run_set(_taskset(), ids=["b", "a"])
    assert [a.task_id for a in attempts] == ["b", "a"]  # order preserved


def test_run_set_rejects_unknown_split():
    runner = AttemptRunner(MockAdapter(responses=["x"]))
    with pytest.raises(KeyError, match="not in taskset"):
        runner.run_set(_taskset(), split="bogus")


def test_attempt_ids_are_unique():
    model = MockAdapter(responses=["x", "y", "z"])
    runner = AttemptRunner(model)
    attempts = runner.run_set(_taskset())
    ids = [a.attempt_id for a in attempts]
    assert len(set(ids)) == len(ids)
