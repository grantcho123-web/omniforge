"""Human-grader stub.

For tasks requiring expert judgment that no auto-grader can replicate
(open-ended financial analysis, language-quality evaluation, cultural
appropriateness, etc.). The full implementation queues attempts to a
reviewer workbench and returns a result asynchronously; this v0 stub
synchronously records the attempt and returns a placeholder result so
pipelines exercising the API don't break.

Config:

* ``queue``: which review queue to enqueue into. Default ``"default"``.
* ``instructions``: rubric or instructions shown to the human reviewer.
* ``placeholder_score``: 0.0 by default. The real workbench overwrites
  this when a human submits a real score.
"""
from __future__ import annotations

import logging

from omniforge.core.grader import Grader, register_grader
from omniforge.core.task import Attempt, GraderSpec, GradingResult, Task

log = logging.getLogger(__name__)


class HumanGrader(Grader):
    name = "human"

    def __init__(self, spec: GraderSpec) -> None:
        cfg = spec.config
        self.queue: str = cfg.get("queue", "default")
        self.instructions: str | None = cfg.get("instructions")
        self.placeholder_score: float = float(cfg.get("placeholder_score", 0.0))

    def grade(self, task: Task, attempt: Attempt) -> GradingResult:
        # In v0, "queueing" is just structured logging. The reviewer
        # workbench (v0.3+) reads these and surfaces them to humans.
        log.info(
            "human_grader.enqueued queue=%s task_id=%s attempt_id=%s",
            self.queue,
            task.metadata.task_id,
            attempt.attempt_id,
        )
        return GradingResult(
            task_id=task.metadata.task_id,
            attempt_id=attempt.attempt_id,
            score=self.placeholder_score,
            passed=False,
            rationale=f"queued for human review on queue={self.queue!r}",
            graded_by=f"human:{self.queue}",
            metadata={"awaiting_review": True, "queue": self.queue},
        )


@register_grader("human")
def _factory(spec: GraderSpec) -> HumanGrader:
    return HumanGrader(spec)
