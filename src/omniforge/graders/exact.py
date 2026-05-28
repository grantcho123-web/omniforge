"""Exact-match grader.

Compares attempt text against an expected answer. Config:

* ``answer``: the expected string. If omitted, falls back to ``task.reference_answer``.
* ``case_sensitive``: default ``False``.
* ``strip``: default ``True`` — strip leading/trailing whitespace before compare.
* ``answer_field``: ``"auto"`` | ``"parsed_answer"`` | ``"raw_response"``.
* ``numeric_tolerance``: if set, both sides are parsed as floats and compared
  with absolute tolerance. The parser tries plain ``float()`` first, then
  falls back to extracting the LAST number from the response — handles
  markdown-wrapped answers like ``**1028**``, comma-grouped numbers like
  ``1,027.83``, currency suffixes like ``958원``, and chain-of-thought style
  responses that put the final answer at the end.
* ``extract_number``: default ``True``. Set to ``False`` to force strict
  ``float(text)`` only (no fallback extraction).
"""
from __future__ import annotations

import re

from omniforge.core.grader import Grader, attempt_text, register_grader
from omniforge.core.task import Attempt, GraderSpec, GradingResult, Task

# Matches signed decimals with optional thousands-separator commas:
#   1027            ✓
#   -1.0            ✓
#   1,027.83        ✓
#   1,000,000       ✓
#   .5              ✗ (require at least one digit before any dot)
_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # comma-grouped (e.g. 1,027.83)
    r"|-?\d+(?:\.\d+)?"                  # plain (e.g. 1028 or -1.0)
)


def _parse_number(text: str, *, extract: bool = True) -> float | None:
    """Best-effort numeric extraction.

    Order:
    1. Plain ``float(text.strip())`` — fast path; handles "958" and "-1.0".
    2. If ``extract`` is True, regex-find every number-shaped substring and
       take the last one (chain-of-thought answers put the conclusion last).
       Strip commas before parsing so "1,027.83" works.
    """
    s = text.strip()
    try:
        return float(s)
    except (ValueError, AttributeError):
        pass
    if not extract:
        return None
    matches = _NUMBER_RE.findall(s)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except (ValueError, AttributeError):
        return None


class ExactMatchGrader(Grader):
    name = "exact_match"

    def __init__(self, spec: GraderSpec) -> None:
        cfg = spec.config
        self.expected: str | None = cfg.get("answer")
        self.case_sensitive: bool = cfg.get("case_sensitive", False)
        self.strip: bool = cfg.get("strip", True)
        self.answer_field: str = cfg.get("answer_field", "auto")
        self.numeric_tolerance: float | None = cfg.get("numeric_tolerance")
        self.extract_number: bool = cfg.get("extract_number", True)

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
            e = _parse_number(expected, extract=self.extract_number)
            g = _parse_number(got, extract=self.extract_number)
            if e is None or g is None:
                preview = got if len(got) <= 80 else got[:77] + "..."
                return False, (
                    f"numeric_tolerance set but unparseable: "
                    f"expected={expected!r} got={preview!r}"
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
