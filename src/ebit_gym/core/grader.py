"""Grader interface and registry.

A ``Grader`` is anything that turns a ``(Task, Attempt)`` into a
``GradingResult``. Reference implementations live in ``ebit_gym.graders``.

The registry lets corpora reference graders by string id in their
``GraderSpec.type`` field, so a ``Task`` can be authored as pure JSON
without importing any grader code.

Construction:

* For graders whose config is fully serializable (``exact_match``,
  ``regex``, ``human``, most ``composite`` setups), ``make_grader(spec)``
  builds the grader from the spec alone.
* For graders that need runtime resources — most importantly the LLM
  judge, which needs a model callable — register the grader class but
  build it manually with the resources injected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ebit_gym.core.task import Attempt, GraderSpec, GradingResult, Task


class Grader(ABC):
    """Abstract scorer. Stateless beyond constructor-time config."""

    name: str = "abstract"

    @abstractmethod
    def grade(self, task: Task, attempt: Attempt) -> GradingResult:
        """Return a GradingResult for this attempt at this task."""


# ---------------------------------------------------------------- registry

_GraderFactory = Callable[[GraderSpec], Grader]
_REGISTRY: dict[str, _GraderFactory] = {}


def register_grader(name: str) -> Callable[[_GraderFactory], _GraderFactory]:
    """Decorator: register a factory under ``name`` so ``make_grader`` can find it.

    The factory takes a ``GraderSpec`` and returns a ``Grader``. Use this for
    graders that can be built from spec alone; for graders that need
    runtime resources (e.g. a model client), construct them directly.
    """

    def deco(factory: _GraderFactory) -> _GraderFactory:
        if name in _REGISTRY:
            raise ValueError(f"grader '{name}' is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def make_grader(spec: GraderSpec) -> Grader:
    """Build a grader from its serialized spec.

    Raises ``KeyError`` if the spec's type is unknown — callers building
    runtime-resourced graders (LLM judge) should construct them directly
    rather than going through this factory.
    """
    if spec.type not in _REGISTRY:
        raise KeyError(
            f"unknown grader type {spec.type!r}; "
            f"known: {sorted(_REGISTRY)} "
            f"(runtime-resourced graders like llm_judge must be constructed directly)"
        )
    return _REGISTRY[spec.type](spec)


def registered_graders() -> list[str]:
    """List registered grader type names. Useful for the CLI and tests."""
    return sorted(_REGISTRY)


# ----------------------------------------------------------- shared helpers


def attempt_text(attempt: Attempt, field: str = "auto") -> str:
    """Pick the text to grade.

    ``field`` can be ``"parsed_answer"``, ``"raw_response"``, or ``"auto"``
    (parsed if present, else raw). Centralized so every grader applies the
    same rule.
    """
    if field == "parsed_answer":
        return attempt.parsed_answer or ""
    if field == "raw_response":
        return attempt.raw_response
    if field == "auto":
        return attempt.parsed_answer if attempt.parsed_answer is not None else attempt.raw_response
    raise ValueError(f"unknown field selector: {field!r}")
