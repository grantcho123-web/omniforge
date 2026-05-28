"""Core platform schemas.

The four primary types:

* ``Task`` — a single unit of work an AI is asked to do, plus the rubric
  for grading attempts. Self-contained and JSON-serializable.
* ``TaskSet`` — a versioned collection of tasks with optional train/eval
  splits and provenance metadata.
* ``Attempt`` — a single model's response to a single task, with cost
  and latency telemetry.
* ``GradingResult`` — the score awarded to an attempt, with rationale and
  optional sub-score breakdown.

These types are intentionally generic. Domain-specific tasks (finance,
language, code, etc.) are *instances* of ``Task``, not subclasses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.2.0"

Difficulty = Literal["easy", "medium", "hard", "expert"]
AttemptProtocol = Literal["one_shot", "multi_turn", "tool_use"]
MaterialKind = Literal["text", "file", "url", "code", "table", "image"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Base(BaseModel):
    """Shared pydantic config: forbid unknown fields, allow attribute access."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------- Task


class TaskMetadata(_Base):
    """Provenance and routing metadata for a task.

    ``domain`` uses a dotted hierarchy so corpora can be filtered cleanly,
    e.g. ``"finance.quant.interview"`` or ``"language.korean.reasoning"``.
    """

    task_id: str = Field(..., description="Stable unique identifier, ideally a UUID or slug.")
    schema_version: str = Field(default=SCHEMA_VERSION)
    version: str = Field(default="0.1.0", description="Semver of this task's content.")
    domain: str = Field(..., description="Dotted domain path, e.g. 'finance.quant'.")
    difficulty: Difficulty = "medium"
    author: str = Field(..., description="Human author name, or 'synthetic:<source>'.")
    created_at: datetime = Field(default_factory=_utcnow)
    tags: list[str] = Field(default_factory=list)
    language: str = Field(
        default="en", description="BCP-47 language tag, e.g. 'ko', 'ja', 'zh-CN'."
    )


class TaskMaterial(_Base):
    """A piece of reference material attached to a task.

    Inline content fits in ``content``; larger payloads (multi-MB tables,
    images) should set ``content`` to a URI and the runner will fetch.
    """

    kind: MaterialKind
    content: str
    mime_type: str | None = None
    name: str | None = None


class GraderSpec(_Base):
    """Configuration declaring how to grade attempts at this task.

    The ``type`` selects a registered Grader implementation; ``config`` is
    that grader's parameter bag. This indirection keeps Task instances
    serializable without coupling to grader code.
    """

    type: str = Field(..., description="Registered grader name, e.g. 'exact_match', 'llm_judge'.")
    config: dict[str, Any] = Field(default_factory=dict)


class Task(_Base):
    """A single unit of work an AI is asked to do."""

    metadata: TaskMetadata
    prompt: str = Field(..., description="The actual instruction the model sees.")
    materials: list[TaskMaterial] = Field(default_factory=list)
    attempt_protocol: AttemptProtocol = "one_shot"
    grader_spec: GraderSpec
    reference_answer: str | None = Field(
        default=None, description="Canonical answer for auto-graders that need it."
    )
    rubric: str | None = Field(
        default=None, description="Free-text rubric for LLM-judge and human graders."
    )

    @field_validator("prompt")
    @classmethod
    def _prompt_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v


# ------------------------------------------------------------------ TaskSet


class TaskSet(_Base):
    """A versioned collection of tasks with optional splits."""

    name: str = Field(..., description="Stable corpus name, e.g. 'finance-quant-v1'.")
    version: str = Field(default="0.1.0")
    schema_version: str = Field(default=SCHEMA_VERSION)
    description: str = ""
    tasks: list[Task] = Field(default_factory=list)
    splits: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Named splits → list of task_ids, e.g. {'train': [...], 'eval': [...]}.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Author orgs, methodology refs, license, build commit, etc.",
    )

    def task_by_id(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.metadata.task_id == task_id:
                return t
        raise KeyError(task_id)

    @field_validator("tasks")
    @classmethod
    def _unique_task_ids(cls, v: list[Task]) -> list[Task]:
        ids = [t.metadata.task_id for t in v]
        dupes = {x for x in ids if ids.count(x) > 1}
        if dupes:
            raise ValueError(f"duplicate task_ids in TaskSet: {sorted(dupes)}")
        return v


# ------------------------------------------------------------------ Attempt


class AttemptCost(_Base):
    """Token + dollar cost of producing an attempt. All fields optional so
    free or self-hosted models can omit them."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None


class Attempt(_Base):
    """A single model's response to a single task."""

    attempt_id: str = Field(..., description="Stable unique identifier for this attempt.")
    task_id: str
    task_version: str = Field(..., description="Version of the task that was attempted.")
    model: str = Field(
        ...,
        description="Provider-qualified model id, e.g. 'anthropic:claude-sonnet-4-5'.",
    )
    raw_response: str
    parsed_answer: str | None = Field(
        default=None, description="Answer extracted from raw_response, if parseable."
    )
    cost: AttemptCost | None = None
    latency_ms: int | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime = Field(default_factory=_utcnow)
    error: str | None = Field(
        default=None, description="If non-null, the attempt failed and raw_response may be partial."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------- GradingResult


class GradingResult(_Base):
    """The score awarded to an attempt."""

    task_id: str
    attempt_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Canonical 0..1 score.")
    passed: bool = Field(..., description="Did the attempt meet the task's pass bar.")
    rationale: str | None = None
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Sub-scores, useful for composite or rubric graders.",
    )
    graded_by: str = Field(
        ..., description="Grader identity, e.g. 'exact_match' or 'human:alice@example.com'."
    )
    graded_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
