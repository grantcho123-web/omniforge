"""Attempt runner.

The runner is the glue between a ``Task``, a ``ModelAdapter``, and the
resulting ``Attempt``. It is intentionally simple: one method per scope
(``run`` for a single task, ``run_set`` for a whole ``TaskSet``), no
threading, no async — those land in a later version once we have a
concrete need.

Design choices:
- The runner does NOT grade. Grading is a separate step so attempts can
  be re-graded with different rubrics without re-running the model.
- Attempts are constructed even on adapter error, with ``error`` set
  and ``raw_response`` empty — failed attempts are data, not exceptions.
- ``attempt_id`` is a UUID4 hex so attempts can be cross-referenced
  across grading runs and storage backends.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from ebit_gym.core.model import ModelAdapter
from ebit_gym.core.task import Attempt, AttemptCost, Task, TaskSet


@dataclass
class RunnerConfig:
    """Knobs that apply to every attempt in a run."""

    system_prompt: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0


class AttemptRunner:
    """Drives a model through a task or a whole task set."""

    def __init__(self, model: ModelAdapter, config: RunnerConfig | None = None) -> None:
        self.model = model
        self.config = config or RunnerConfig()

    def run(self, task: Task) -> Attempt:
        """Produce one attempt for one task."""
        attempt_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc)

        prompt = self._render_prompt(task)
        resp = self.model.call(
            prompt,
            system=self.config.system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        completed_at = datetime.now(timezone.utc)

        cost = None
        if resp.input_tokens is not None or resp.output_tokens is not None or resp.usd is not None:
            cost = AttemptCost(
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                usd=resp.usd,
            )

        return Attempt(
            attempt_id=attempt_id,
            task_id=task.metadata.task_id,
            task_version=task.metadata.version,
            model=self.model.name,
            raw_response=resp.raw_response,
            cost=cost,
            latency_ms=resp.latency_ms,
            started_at=started_at,
            completed_at=completed_at,
            error=resp.error,
            metadata=dict(resp.metadata),
        )

    def run_set(
        self,
        taskset: TaskSet,
        split: str | None = None,
        ids: list[str] | None = None,
    ) -> list[Attempt]:
        """Produce one attempt per task in the set.

        Filtering precedence: explicit ``ids`` wins, then ``split``, then all tasks.
        """
        tasks: list[Task]
        if ids is not None:
            tasks = [taskset.task_by_id(t) for t in ids]
        elif split is not None:
            if split not in taskset.splits:
                raise KeyError(
                    f"split {split!r} not in taskset; "
                    f"known splits: {sorted(taskset.splits)}"
                )
            tasks = [taskset.task_by_id(t) for t in taskset.splits[split]]
        else:
            tasks = list(taskset.tasks)

        return [self.run(t) for t in tasks]

    # ----------------------------------------------------------- prompt rendering

    @staticmethod
    def _render_prompt(task: Task) -> str:
        """Render the model-visible prompt from a task.

        v0 rendering: prompt followed by inline materials. Tool-using
        agent loops (v0.3+) will replace this with a much richer
        message construction.
        """
        if not task.materials:
            return task.prompt

        parts: list[str] = [task.prompt, ""]
        for i, m in enumerate(task.materials, start=1):
            header = f"--- material {i}"
            if m.name:
                header += f" ({m.name})"
            header += " ---"
            parts.append(header)
            parts.append(m.content)
        return "\n".join(parts)
