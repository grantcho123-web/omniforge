"""Regex grader.

Passes when the configured pattern matches the attempt text. Config:

* ``pattern``: required regex.
* ``flags``: optional, subset of ``"imsx"`` — ``i`` ignorecase, ``m`` multiline,
  ``s`` dotall, ``x`` verbose.
* ``must_match``: default ``True``. If ``False``, the pattern must NOT match
  to pass — useful for "the answer must not mention X" rules.
* ``answer_field``: ``"auto"`` | ``"parsed_answer"`` | ``"raw_response"``.
"""
from __future__ import annotations

import re

from ebit_gym.core.grader import Grader, attempt_text, register_grader
from ebit_gym.core.task import Attempt, GraderSpec, GradingResult, Task

_FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


def _parse_flags(s: str) -> int:
    out = 0
    for ch in s:
        if ch not in _FLAG_MAP:
            raise ValueError(f"unknown regex flag {ch!r}; allowed: imsx")
        out |= _FLAG_MAP[ch]
    return out


class RegexGrader(Grader):
    name = "regex"

    def __init__(self, spec: GraderSpec) -> None:
        cfg = spec.config
        pattern = cfg.get("pattern")
        if not pattern:
            raise ValueError("regex grader requires 'pattern' in config")
        self.pattern = re.compile(pattern, _parse_flags(cfg.get("flags", "")))
        self.must_match: bool = cfg.get("must_match", True)
        self.answer_field: str = cfg.get("answer_field", "auto")

    def grade(self, task: Task, attempt: Attempt) -> GradingResult:
        text = attempt_text(attempt, self.answer_field)
        matched = self.pattern.search(text) is not None
        ok = matched if self.must_match else (not matched)
        rationale = (
            f"pattern {'matched' if matched else 'did not match'}; "
            f"must_match={self.must_match}"
        )
        return GradingResult(
            task_id=task.metadata.task_id,
            attempt_id=attempt.attempt_id,
            score=1.0 if ok else 0.0,
            passed=ok,
            rationale=rationale,
            graded_by=self.name,
        )


@register_grader("regex")
def _factory(spec: GraderSpec) -> RegexGrader:
    return RegexGrader(spec)
