"""Exact-match grader.

Compares attempt text against an expected answer. Config:

* ``answer``: the expected string. If omitted, falls back to ``task.reference_answer``.
* ``case_sensitive``: default ``False``.
* ``strip``: default ``True`` — strip leading/trailing whitespace before compare.
* ``answer_field``: ``"auto"`` | ``"parsed_answer"`` | ``"raw_response"``.
* ``numeric_tolerance``: if set, both sides are parsed as floats and compared
  with absolute tolerance. Useful for quant questions where ``"6.0"`` and
  ``"6"`` should match.
"""
from __future__ import annotations

from omniforge.core.grader import Grader, attempt_text, register_grader
from omniforge.core.task import Attempt, GraderSpec, GradingResult, Task


class ExactMatchGrader(Grader):
    name = "exact_match"

    def __init__(self, spec: GraderSpec) -> None:
        cfg = spec.config
        self.expected: str | None = cfg.get("answer")
        self.case_sensitive: bool = cfg.get("case_sensitive", False)
        self.strip: bool = cfg.get("strip", True)
        self.answer_field: str = cfg.get("answer_field", "auto")
        self.numeric_tolerance: float | None = cfg.get("numeric_tolerance")

    def grade(self, task: Task, attempt: Attempt) -> GradingResult:
        expected = self.expected if self.expected is not None else task.reference_answer
        if expected is None:
            raise ValueError(
                f"exact_match grader for task {task.metadata.task_id!r} has no answer "
                f"in config and task.reference_answer is also unset"
            )

        got = attempt_text(attempt, self.answer_field)
        match, rationale = self._compare(expected, got)
        return GradingResult(
            task_id=task.metadata.task_id,
            attempt_id=attempt.attempt_id,
            score=1.0 if match else 0.0,
            passed=match,
            rationale=rationale,
            graded_by=self.name,
        )

    def _compare(self, expected: str, got: str) -> tuple[bool, str]:
        if self.numeric_tolerance is not None:
            try:
                e = float(expected.strip())
                g = float(got.strip())
            except (ValueError, AttributeError):
                return False, (
                    f"numeric_tolerance set but inputs unparseable: "
                    f"{expected!r} vs {got!r}"
                )
            ok = abs(e - g) <= self.numeric_tolerance
            return ok, f"|{e} - {g}| = {abs(e - g):.6g} (tol {self.numeric_tolerance})"

        e = expected
        g = got
        if self.strip:
            e = e.strip()
            g = g.strip()
        if not self.case_sensitive:
            e = e.lower()
            g = g.lower()
        ok = e == g
        return ok, ("exact match" if ok else f"expected {e!r}, got {g!r}")


@register_grader("exact_match")
def _factory(spec: GraderSpec) -> ExactMatchGrader:
    return ExactMatchGrader(spec)
