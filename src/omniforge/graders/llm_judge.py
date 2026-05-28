"""LLM-as-judge grader.

Uses another LLM to score an attempt against a rubric. Config:

* ``rubric``: the rubric text. Falls back to ``task.rubric`` if unset.
* ``pass_threshold``: default ``0.7`` — score >= threshold counts as passed.
* ``judge_model``: informational only here (e.g. ``"anthropic:claude-4.6-sonnet"``);
  the actual judge callable is injected at construction time.

Construction:

    judge = LLMJudgeGrader(spec, judge_fn=my_callable)

where ``judge_fn`` has signature ``(prompt: str) -> str`` and returns the
judge model's raw response. ``LLMJudgeGrader`` parses the JSON
``{"score": float, "rationale": str}`` out of that response.

This grader is intentionally not registered with ``make_grader`` — it
requires a runtime callable. Wire it up explicitly in your eval pipeline.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from omniforge.core.grader import Grader, attempt_text
from omniforge.core.task import Attempt, GraderSpec, GradingResult, Task

JudgeFn = Callable[[str], str]


_JUDGE_PROMPT_TEMPLATE = """You are evaluating an AI's response to a task.

TASK PROMPT:
{prompt}

{reference_block}RUBRIC:
{rubric}

AI'S RESPONSE:
{response}

Score the response from 0.0 (totally wrong / off-topic / harmful) to 1.0
(perfect according to the rubric). Be calibrated: 0.5 is "partially correct
or missing key elements", 0.8 is "correct with minor issues", 1.0 is
"correct and complete".

Respond ONLY in JSON of the form:
{{"score": <float between 0 and 1>, "rationale": "<one or two sentences>"}}
"""


_JSON_RE = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)


class LLMJudgeGrader(Grader):
    name = "llm_judge"

    def __init__(self, spec: GraderSpec, judge_fn: JudgeFn) -> None:
        cfg = spec.config
        self.rubric: str | None = cfg.get("rubric")
        self.pass_threshold: float = float(cfg.get("pass_threshold", 0.7))
        self.judge_model: str = cfg.get("judge_model", "unknown")
        self.answer_field: str = cfg.get("answer_field", "raw_response")
        self.judge_fn = judge_fn

    def grade(self, task: Task, attempt: Attempt) -> GradingResult:
        rubric = self.rubric if self.rubric is not None else task.rubric
        if not rubric:
            raise ValueError(
                f"llm_judge grader for task {task.metadata.task_id!r} has no rubric "
                f"in config and task.rubric is also unset"
            )

        reference_block = ""
        if task.reference_answer:
            reference_block = f"REFERENCE ANSWER:\n{task.reference_answer}\n\n"

        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            prompt=task.prompt,
            reference_block=reference_block,
            rubric=rubric,
            response=attempt_text(attempt, self.answer_field),
        )

        raw = self.judge_fn(prompt)
        score, rationale, error = self._parse_judge_output(raw)

        return GradingResult(
            task_id=task.metadata.task_id,
            attempt_id=attempt.attempt_id,
            score=score,
            passed=score >= self.pass_threshold,
            rationale=rationale,
            graded_by=f"llm_judge:{self.judge_model}",
            metadata={"raw_judge_output": raw, "parse_error": error},
        )

    @staticmethod
    def _parse_judge_output(raw: str) -> tuple[float, str, str | None]:
        """Best-effort extraction of {score, rationale} from the judge output."""
        text = raw.strip()
        # First, try clean JSON.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = _JSON_RE.search(text)
            if not m:
                return 0.0, "judge output not parseable as JSON", "no JSON found"
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                return 0.0, f"judge JSON malformed: {e}", str(e)

        try:
            score = float(data["score"])
        except (KeyError, TypeError, ValueError) as e:
            return 0.0, "judge output missing/invalid 'score'", str(e)
        score = max(0.0, min(1.0, score))
        rationale = str(data.get("rationale", ""))[:1000]
        return score, rationale, None
