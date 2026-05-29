"""Exact-match grader.

Compares attempt text against an expected answer. Config:

* ``answer``: the expected string. If omitted, falls back to ``task.reference_answer``.
* ``case_sensitive``: default ``False``.
* ``strip``: default ``True`` — strip leading/trailing whitespace before compare.
* ``answer_field``: ``"auto"`` | ``"parsed_answer"`` | ``"raw_response"``.
* ``numeric_tolerance``: if set, both sides are parsed as floats and compared
  with absolute tolerance.
* ``extract_answer``: default ``True``. When True, before comparison the
  grader extracts the model's stated final answer using a layered heuristic
  (``\\boxed{X}`` LaTeX > ``**X**`` markdown bold > last non-empty line, with
  "Answer:" / "Final answer:" prefixes stripped). This works for both
  numeric and non-numeric answers — fixes both "prose after the answer
  swallows the wrong number" failures and "letter-answer in markdown bold
  loses full-string compare" failures.
* ``extract_number``: backward-compat alias for ``extract_answer`` (kept for
  pre-existing configs).
"""
from __future__ import annotations

import re

from omniforge.core.grader import Grader, attempt_text, register_grader
from omniforge.core.task import Attempt, GraderSpec, GradingResult, Task

# Signed decimals with optional thousands-separator commas:
#   1027            ✓
#   -1.0            ✓
#   1,027.83        ✓
#   1,000,000       ✓
#   .5              ✗ (require at least one digit before any dot)
_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # comma-grouped (e.g. 1,027.83)
    r"|-?\d+(?:\.\d+)?"                  # plain (e.g. 1028 or -1.0)
)
# LaTeX \boxed{...} — common in math-tutoring style responses.
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
# Markdown bold **...**  (line-bounded, no embedded *).
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
# Strip "Answer:" / "Final answer:" / "The answer is" prefixes from a candidate.
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+)?[:=]?\s*",
    flags=re.IGNORECASE,
)


def _extract_final_answer(text: str) -> str | None:
    """Extract the model's stated final answer from a chain-of-thought response.

    Layered heuristic, in priority order:
      1. The LAST ``\\boxed{X}`` expression (LaTeX answer convention).
      2. The LAST ``**X**`` markdown bold block, with "Answer:" prefix stripped.
      3. The LAST non-empty line of the response, with prefix stripped.

    Returns the candidate string (trimmed, trailing ``.`` removed), or ``None``
    if the input is empty. The caller is responsible for any further parsing
    (e.g. as a number for numeric_tolerance comparison, or a single letter
    for multiple-choice).

    Why this design: real LLM responses to math/MCQ tasks consistently put the
    final answer either in ``\\boxed{}``, in ``**bold**``, or alone on the
    last line. They often continue with prose explanation AFTER the answer,
    which breaks naive "last number in text" extractors.
    """
    s = text.strip()
    if not s:
        return None

    # 1. \boxed{X}
    boxed = _BOXED_RE.findall(s)
    if boxed:
        return boxed[-1].strip().rstrip(".")

    # 2. **bold** — pick the last occurrence
    bolds = _BOLD_RE.findall(s)
    if bolds:
        candidate = _ANSWER_PREFIX_RE.sub("", bolds[-1].strip())
        return candidate.strip().rstrip(".")

    # 3. Last non-empty line
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        candidate = _ANSWER_PREFIX_RE.sub("", lines[-1])
        return candidate.strip().rstrip(".")

    return None


def _parse_number(text: str, *, extract: bool = True) -> float | None:
    """Best-effort numeric extraction.

    Order:
      1. Plain ``float(text.strip())`` — fast path for clean responses.
      2. If ``extract`` is True, run ``_extract_final_answer`` and try to
         parse THAT candidate as a number (handles bolded answers and
         responses with prose after the answer).
      3. If still not a number, scan the candidate for any number-shaped
         substring and take the last one (handles "Answer: 1,027.83 dollars"
         style responses).
      4. Last-resort fallback: scan the whole text and take the last number
         anywhere. Less reliable; used only when extraction fails.
    """
    s = text.strip()
    try:
        return float(s)
    except (ValueError, AttributeError):
        pass
    if not extract:
        return None

    # Try the final-answer extractor first.
    final = _extract_final_answer(text)
    if final is not None:
        candidate = final.replace(",", "").strip()
        try:
            return float(candidate)
        except (ValueError, AttributeError):
            pass
        # Maybe the candidate is "Answer: 42." or "1,027.83 dollars" — pull
        # the last number out of the candidate (not the whole text).
        matches = _NUMBER_RE.findall(final)
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except (ValueError, AttributeError):
                pass

    # Last resort: scan whole text.
    matches = _NUMBER_RE.findall(text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except (ValueError, AttributeError):
            pass
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
        # Unified extract flag; extract_number kept as alias for back-compat.
        self.extract_answer: bool = cfg.get(
            "extract_answer", cfg.get("extract_number", True)
        )

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
            e = _parse_number(expected, extract=self.extract_answer)
            g = _parse_number(got, extract=self.extract_answer)
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
        # Extract final answer for non-numeric comparison too. Prevents the
        # "Sonnet wrote a 200-word analysis ending in **C**" failure mode.
        if self.extract_answer:
            extracted = _extract_final_answer(g)
            if extracted is not None:
                g = extracted
        if not self.case_sensitive:
            e = e.lower()
            g = g.lower()
        ok = e == g
        return ok, ("exact match" if ok else f"expected {e!r}, got {g!r}")


@register_grader("exact_match")
def _factory(spec: GraderSpec) -> ExactMatchGrader:
    return ExactMatchGrader(spec)
