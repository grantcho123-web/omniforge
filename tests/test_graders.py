"""Tests for the grader interface and reference implementations."""
from __future__ import annotations

import logging

import pytest

# Importing this registers all built-in graders.
import omniforge.graders  # noqa: F401
from omniforge.core import Attempt, GraderSpec, Task, TaskMetadata
from omniforge.core.grader import make_grader, registered_graders
from omniforge.graders.composite import CompositeGrader
from omniforge.graders.exact import ExactMatchGrader
from omniforge.graders.llm_judge import LLMJudgeGrader
from omniforge.graders.regex import RegexGrader


def _task(
    task_id: str = "t-1",
    reference_answer: str | None = "6",
    rubric: str | None = None,
    grader_spec: GraderSpec | None = None,
) -> Task:
    return Task(
        metadata=TaskMetadata(task_id=task_id, domain="x", author="a"),
        prompt="...",
        grader_spec=grader_spec or GraderSpec(type="exact_match"),
        reference_answer=reference_answer,
        rubric=rubric,
    )


def _attempt(
    raw: str = "6",
    parsed: str | None = None,
    task_id: str = "t-1",
    attempt_id: str = "a-1",
) -> Attempt:
    return Attempt(
        attempt_id=attempt_id,
        task_id=task_id,
        task_version="0.1.0",
        model="anthropic:claude-sonnet-4-5",
        raw_response=raw,
        parsed_answer=parsed,
    )


# ----------------------------------------------------------------- registry


def test_registry_lists_all_builtins():
    names = registered_graders()
    assert {"exact_match", "regex", "composite", "human"}.issubset(names)
    # llm_judge needs a runtime callable so it's deliberately NOT registered.
    assert "llm_judge" not in names


def test_make_grader_rejects_unknown_type():
    with pytest.raises(KeyError, match="unknown grader"):
        make_grader(GraderSpec(type="not_real"))


def test_make_grader_returns_correct_class():
    g = make_grader(GraderSpec(type="exact_match", config={"answer": "x"}))
    assert isinstance(g, ExactMatchGrader)


# --------------------------------------------------------------- exact_match


def test_exact_match_basic_pass():
    g = make_grader(GraderSpec(type="exact_match", config={"answer": "6"}))
    r = g.grade(_task(), _attempt(raw="6"))
    assert r.passed
    assert r.score == 1.0


def test_exact_match_falls_back_to_reference_answer():
    g = make_grader(GraderSpec(type="exact_match"))
    r = g.grade(_task(reference_answer="42"), _attempt(raw="42"))
    assert r.passed


def test_exact_match_case_insensitive_by_default():
    g = make_grader(GraderSpec(type="exact_match", config={"answer": "Yes"}))
    r = g.grade(_task(reference_answer=None), _attempt(raw="YES"))
    assert r.passed


def test_exact_match_case_sensitive_when_configured():
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "Yes", "case_sensitive": True})
    )
    r = g.grade(_task(reference_answer=None), _attempt(raw="yes"))
    assert not r.passed


def test_exact_match_numeric_tolerance():
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "6.0", "numeric_tolerance": 0.01})
    )
    assert g.grade(_task(), _attempt(raw="6.005")).passed
    assert not g.grade(_task(), _attempt(raw="6.5")).passed


def test_exact_match_uses_parsed_answer_when_present():
    g = make_grader(GraderSpec(type="exact_match", config={"answer": "6"}))
    r = g.grade(_task(), _attempt(raw="The answer is six", parsed="6"))
    assert r.passed


def test_exact_match_raises_when_no_answer_available():
    g = make_grader(GraderSpec(type="exact_match"))
    with pytest.raises(ValueError, match="no answer"):
        g.grade(_task(reference_answer=None), _attempt())


def test_exact_match_extracts_number_from_markdown_response():
    """Real-world LLM responses wrap answers in markdown and prose; the
    grader extracts the last number from the response when plain float()
    fails."""
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "1027", "numeric_tolerance": 2.0})
    )
    response = (
        "# 채권 현재가치 계산\n\nPV = 60/1.05 + 60/1.1025 + 1060/1.1576\n"
        "PV = 57.14 + 54.42 + 916.27\nPV = 1,027.83\n\n**1028**"
    )
    r = g.grade(_task(), _attempt(raw=response))
    assert r.passed
    assert r.score == 1.0


def test_exact_match_extracts_comma_grouped_numbers():
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "1027", "numeric_tolerance": 1.0})
    )
    # Last number is "1,027.83" — should parse to 1027.83 after comma strip,
    # within tolerance of 1027.
    r = g.grade(_task(), _attempt(raw="The present value comes to 1,027.83 dollars."))
    assert r.passed


def test_exact_match_extracts_negative_numbers():
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "-1.0", "numeric_tolerance": 0.1})
    )
    # Trading-style response that picks short
    r = g.grade(
        _task(reference_answer="-1.0"),
        _attempt(raw="Trend looks bearish so I'd short here.\n\n**-1**"),
    )
    assert r.passed


def test_exact_match_extract_picks_last_number():
    """When multiple numbers appear, the last one wins — matches the
    convention that CoT answers end with the final answer."""
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "42", "numeric_tolerance": 0.5})
    )
    response = "Started with 100, halved to 50, gave away 8. Final answer: 42."
    assert g.grade(_task(), _attempt(raw=response)).passed


def test_exact_match_extract_disabled_keeps_strict_behavior():
    """Setting extract_number=False reverts to the brittle behavior. Useful
    when you want to penalize unformatted responses."""
    g = make_grader(
        GraderSpec(
            type="exact_match",
            config={
                "answer": "1027",
                "numeric_tolerance": 2.0,
                "extract_number": False,
            },
        )
    )
    r = g.grade(_task(), _attempt(raw="**1028**"))
    assert not r.passed
    assert "unparseable" in (r.rationale or "")


def test_exact_match_empty_response_still_fails():
    """The fix shouldn't accidentally pass empty responses."""
    g = make_grader(
        GraderSpec(type="exact_match", config={"answer": "5", "numeric_tolerance": 0.5})
    )
    r = g.grade(_task(), _attempt(raw=""))
    assert not r.passed
    assert "unparseable" in (r.rationale or "")


# --------------------------------------------------------------------- regex


def test_regex_basic_match():
    g = make_grader(GraderSpec(type="regex", config={"pattern": r"\b42\b"}))
    assert g.grade(_task(), _attempt(raw="the answer is 42!")).passed
    assert not g.grade(_task(), _attempt(raw="forty-two")).passed


def test_regex_flags_ignorecase():
    g = make_grader(GraderSpec(type="regex", config={"pattern": r"hello", "flags": "i"}))
    assert g.grade(_task(), _attempt(raw="HELLO world")).passed


def test_regex_must_not_match():
    g = make_grader(
        GraderSpec(type="regex", config={"pattern": r"forbidden", "must_match": False})
    )
    assert g.grade(_task(), _attempt(raw="clean answer")).passed
    assert not g.grade(_task(), _attempt(raw="contains forbidden words")).passed


def test_regex_rejects_unknown_flag():
    with pytest.raises(ValueError, match="unknown regex flag"):
        RegexGrader(GraderSpec(type="regex", config={"pattern": "x", "flags": "z"}))


def test_regex_requires_pattern():
    with pytest.raises(ValueError, match="requires 'pattern'"):
        RegexGrader(GraderSpec(type="regex"))


# ----------------------------------------------------------------- llm_judge


class _MockJudge:
    """Replays canned judge responses."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def test_llm_judge_parses_clean_json():
    judge = _MockJudge('{"score": 0.85, "rationale": "Mostly correct, minor issue."}')
    g = LLMJudgeGrader(
        GraderSpec(type="llm_judge", config={"rubric": "Is it correct?"}),
        judge_fn=judge,
    )
    r = g.grade(_task(rubric=None), _attempt(raw="my response"))
    assert r.score == 0.85
    assert r.passed
    assert "Mostly correct" in (r.rationale or "")
    assert "Is it correct?" in judge.calls[0]
    assert "my response" in judge.calls[0]


def test_llm_judge_extracts_json_from_chatter():
    judge = _MockJudge(
        'Here is my evaluation:\n{"score": 0.3, "rationale": "Missed the point."}\nDone.'
    )
    g = LLMJudgeGrader(
        GraderSpec(type="llm_judge", config={"rubric": "x", "pass_threshold": 0.5}),
        judge_fn=judge,
    )
    r = g.grade(_task(), _attempt())
    assert r.score == 0.3
    assert not r.passed


def test_llm_judge_clamps_out_of_range_scores():
    judge = _MockJudge('{"score": 7.0, "rationale": "Off the chart."}')
    g = LLMJudgeGrader(
        GraderSpec(type="llm_judge", config={"rubric": "x"}), judge_fn=judge
    )
    r = g.grade(_task(), _attempt())
    assert r.score == 1.0


def test_llm_judge_handles_unparseable_output():
    judge = _MockJudge("The response is roughly OK, maybe a 7 out of 10.")
    g = LLMJudgeGrader(
        GraderSpec(type="llm_judge", config={"rubric": "x"}), judge_fn=judge
    )
    r = g.grade(_task(), _attempt())
    assert r.score == 0.0
    assert not r.passed
    assert "not parseable" in (r.rationale or "")


def test_llm_judge_falls_back_to_task_rubric():
    judge = _MockJudge('{"score": 0.5, "rationale": "ok"}')
    g = LLMJudgeGrader(GraderSpec(type="llm_judge"), judge_fn=judge)
    g.grade(_task(rubric="Task-attached rubric."), _attempt())
    assert "Task-attached rubric." in judge.calls[0]


def test_llm_judge_raises_when_no_rubric_anywhere():
    judge = _MockJudge("{}")
    g = LLMJudgeGrader(GraderSpec(type="llm_judge"), judge_fn=judge)
    with pytest.raises(ValueError, match="no rubric"):
        g.grade(_task(rubric=None), _attempt())


def test_llm_judge_not_in_registry():
    """Intentional design choice — judges need runtime resources."""
    assert "llm_judge" not in registered_graders()


# ---------------------------------------------------------------- composite


def test_composite_weighted_average():
    spec = GraderSpec(
        type="composite",
        config={
            "components": [
                {"weight": 1, "spec": {"type": "exact_match", "config": {"answer": "6"}}},
                {"weight": 3, "spec": {"type": "regex", "config": {"pattern": r"^6$"}}},
            ],
            "pass_threshold": 0.7,
        },
    )
    g = make_grader(spec)
    r = g.grade(_task(), _attempt(raw="6"))
    # Both pass → score is 1.0
    assert r.score == 1.0
    assert r.passed
    assert len(r.breakdown) == 2


def test_composite_partial_credit():
    spec = GraderSpec(
        type="composite",
        config={
            "components": [
                {"weight": 1, "spec": {"type": "exact_match", "config": {"answer": "6"}}},
                {"weight": 1, "spec": {"type": "regex", "config": {"pattern": r"answer"}}},
            ],
            "pass_threshold": 0.7,
        },
    )
    g = make_grader(spec)
    # Exact match fails (raw="hello"), regex fails too.
    r1 = g.grade(_task(), _attempt(raw="hello"))
    assert r1.score == 0.0
    # Exact match fails, regex passes → 0.5 → below threshold
    r2 = g.grade(_task(), _attempt(raw="the answer is wrong"))
    assert r2.score == pytest.approx(0.5)
    assert not r2.passed


def test_composite_from_components_bypasses_registry():
    """For composites including an llm_judge or other runtime-resourced graders."""
    sub_exact = ExactMatchGrader(GraderSpec(type="exact_match", config={"answer": "x"}))
    sub_regex = RegexGrader(GraderSpec(type="regex", config={"pattern": "x"}))
    g = CompositeGrader.from_components([(2.0, sub_exact), (1.0, sub_regex)])
    r = g.grade(_task(reference_answer="x"), _attempt(raw="x"))
    assert r.score == 1.0


def test_composite_rejects_empty_components():
    with pytest.raises(ValueError, match="requires 'components'"):
        CompositeGrader(GraderSpec(type="composite"))


def test_composite_rejects_zero_weight():
    with pytest.raises(ValueError, match="positive"):
        CompositeGrader.from_components(
            [(0.0, ExactMatchGrader(GraderSpec(type="exact_match", config={"answer": "x"})))]
        )


# ------------------------------------------------------------------- human


def test_human_grader_returns_placeholder():
    spec = GraderSpec(
        type="human", config={"queue": "finance-reviewers", "instructions": "Be thorough."}
    )
    g = make_grader(spec)
    r = g.grade(_task(), _attempt())
    assert not r.passed
    assert r.score == 0.0
    assert r.metadata["awaiting_review"] is True
    assert r.metadata["queue"] == "finance-reviewers"
    assert "queued for human review" in (r.rationale or "")


def test_human_grader_logs_enqueue(caplog):
    spec = GraderSpec(type="human", config={"queue": "q1"})
    g = make_grader(spec)
    with caplog.at_level(logging.INFO):
        g.grade(_task(), _attempt())
    assert any("human_grader.enqueued" in rec.message for rec in caplog.records)
