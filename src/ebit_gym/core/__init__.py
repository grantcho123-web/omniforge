"""Platform core: schemas and abstract interfaces.

This subpackage defines the domain-agnostic types that all task corpora,
graders, model adapters, and runners speak. Everything else in ebit-gym
either produces or consumes these types.

Stability promise: schemas are versioned and round-trip JSON-safe so
saved corpora and recorded attempts remain readable across releases.
"""

from ebit_gym.core.task import (
    Attempt,
    AttemptCost,
    GraderSpec,
    GradingResult,
    Task,
    TaskMaterial,
    TaskMetadata,
    TaskSet,
)

__all__ = [
    "Attempt",
    "AttemptCost",
    "GraderSpec",
    "GradingResult",
    "Task",
    "TaskMaterial",
    "TaskMetadata",
    "TaskSet",
]
