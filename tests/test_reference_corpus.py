"""Smoke tests for the shipped reference corpus.

The corpus is regenerated from scripts/build_reference_corpus.py; these
tests guard against:
- Schema drift breaking the committed manifest
- Tasks losing their grader registry references
- The Korean / Japanese content getting mangled by encoding mishaps
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Importing graders registers them so make_grader works.
import omniforge.graders  # noqa: F401
from omniforge.core.grader import make_grader
from omniforge.core.task import TaskSet

MANIFEST = Path(__file__).resolve().parents[1] / "corpora" / "reference-v0" / "manifest.json"


def test_manifest_exists():
    assert MANIFEST.exists(), (
        "Reference corpus manifest is missing. Run "
        "`python scripts/build_reference_corpus.py` from the project root."
    )


@pytest.fixture(scope="module")
def taskset() -> TaskSet:
    return TaskSet.model_validate_json(MANIFEST.read_text(encoding="utf-8"))


def test_corpus_has_expected_size(taskset):
    assert len(taskset.tasks) == 10
    assert taskset.name == "omniforge-reference-v0"


def test_all_task_ids_unique(taskset):
    # TaskSet validator catches dupes at construction, but assert explicitly
    # to harden against future refactors.
    ids = [t.metadata.task_id for t in taskset.tasks]
    assert len(ids) == len(set(ids))


def test_corpus_has_korean_beachhead_coverage(taskset):
    """The GTM premise depends on Korean-language coverage being present
    and substantial."""
    korean_tasks = [t for t in taskset.tasks if t.metadata.language == "ko"]
    assert len(korean_tasks) >= 3
    # At least one numeric (auto-graded) and at least one open-ended (human-graded).
    assert any(t.grader_spec.type in {"exact_match", "regex"} for t in korean_tasks)
    assert any(t.grader_spec.type == "human" for t in korean_tasks)


def test_corpus_exercises_every_grader_type(taskset):
    types = {t.grader_spec.type for t in taskset.tasks}
    # llm_judge is intentionally not in the reference set since it needs runtime.
    assert "exact_match" in types
    assert "regex" in types
    assert "composite" in types
    assert "human" in types


def test_every_auto_grader_resolves(taskset):
    """Every non-runtime grader in the corpus must build via the registry."""
    for task in taskset.tasks:
        if task.grader_spec.type == "llm_judge":
            continue
        grader = make_grader(task.grader_spec)
        assert grader is not None
        assert hasattr(grader, "grade")


def test_corpus_splits_reference_real_ids(taskset):
    all_ids = {t.metadata.task_id for t in taskset.tasks}
    for split_name, ids in taskset.splits.items():
        for task_id in ids:
            assert task_id in all_ids, (
                f"split {split_name!r} references unknown task {task_id!r}"
            )


def test_korean_content_round_trips_unmangled(taskset):
    """Native Korean characters must survive load/parse without escape mangling."""
    korean_task = next(
        t for t in taskset.tasks if t.metadata.task_id == "q-ko-culture-001"
    )
    assert "한국" in korean_task.prompt  # native characters present
    assert "명절" in korean_task.prompt


def test_japanese_content_present(taskset):
    ja_task = next(t for t in taskset.tasks if t.metadata.language == "ja")
    assert "円" in ja_task.prompt  # yen symbol
