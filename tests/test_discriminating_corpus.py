"""Smoke tests for the discriminating-v0 corpus.

Unlike test_reference_corpus.py (which guards the small demo corpus),
these tests focus on the structural correctness of the harder corpus —
it should be appreciably bigger than reference-v0 and exercise multiple
domains.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Importing graders registers them so make_grader works.
import omniforge.graders  # noqa: F401
from omniforge.core.grader import make_grader
from omniforge.core.task import TaskSet

MANIFEST = (
    Path(__file__).resolve().parents[1] / "corpora" / "discriminating-v0" / "manifest.json"
)


def test_manifest_exists():
    assert MANIFEST.exists(), (
        "Discriminating corpus manifest is missing. Run "
        "`python scripts/build_discriminating_corpus.py` from the project root."
    )


@pytest.fixture(scope="module")
def taskset() -> TaskSet:
    return TaskSet.model_validate_json(MANIFEST.read_text(encoding="utf-8"))


def test_corpus_size_and_name(taskset):
    assert taskset.name == "omniforge-discriminating-v0"
    assert len(taskset.tasks) == 15


def test_corpus_has_all_four_domain_buckets(taskset):
    """Splits should expose finance / puzzles / probability / multilingual
    as named slices."""
    for split in ("finance", "puzzles", "probability", "multilingual"):
        assert split in taskset.splits, f"missing split: {split}"
        assert len(taskset.splits[split]) >= 3


def test_corpus_languages_diverse(taskset):
    """Multilingual coverage is part of the design — at least one task per
    target language."""
    langs = {t.metadata.language for t in taskset.tasks}
    assert "en" in langs
    assert "ko" in langs
    assert "ja" in langs
    assert "zh-CN" in langs


def test_every_task_has_reference_answer(taskset):
    """A discriminating corpus must be auto-gradable — every task needs an
    answer to compare against."""
    for task in taskset.tasks:
        assert task.reference_answer, (
            f"task {task.metadata.task_id} missing reference_answer"
        )


def test_every_grader_resolves(taskset):
    for task in taskset.tasks:
        grader = make_grader(task.grader_spec)
        assert grader is not None


def test_numeric_tasks_have_tolerance(taskset):
    """Any task whose reference answer is a number should set
    numeric_tolerance — otherwise prose-wrapped responses fail."""
    for task in taskset.tasks:
        if task.grader_spec.type != "exact_match":
            continue
        try:
            float(task.reference_answer)
        except (TypeError, ValueError):
            # Non-numeric answers (e.g. multiple-choice letter) don't need tolerance.
            continue
        # Numeric answer → must have tolerance, otherwise grader is brittle.
        assert "numeric_tolerance" in task.grader_spec.config, (
            f"task {task.metadata.task_id} has numeric reference_answer "
            f"{task.reference_answer!r} but no numeric_tolerance — will fail "
            "any model that explains its work."
        )


def test_korean_content_round_trips(taskset):
    ko = next(t for t in taskset.tasks if t.metadata.language == "ko")
    assert "한국" in ko.prompt or "회사" in ko.prompt


def test_japanese_and_chinese_present(taskset):
    ja = next(t for t in taskset.tasks if t.metadata.language == "ja")
    assert "ビジネス" in ja.prompt
    zh = next(t for t in taskset.tasks if t.metadata.language == "zh-CN")
    assert "净利润" in zh.prompt or "市盈率" in zh.prompt
