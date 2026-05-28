"""Model adapter interface.

A ``ModelAdapter`` is a thin, uniform wrapper around an LLM provider's
chat API. The adapter does not know about tasks — it just answers
prompts. The attempt runner wraps the response with task context to
produce a full ``Attempt``.

This separation lets the same adapter serve many roles: generating
attempts, judging in an LLM-judge grader, generating synthetic tasks,
etc.

Registry: adapters self-register under their provider:model id (e.g.,
``"anthropic:claude-4.6-sonnet"``) so the CLI can look them up by string.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """A model's response to a single prompt, with telemetry.

    All token/cost fields are optional: free models, mock adapters, and
    self-hosted endpoints often can't or don't report them.
    """

    raw_response: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    latency_ms: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Provider-agnostic LLM client."""

    #: Provider-qualified model id, e.g. ``"anthropic:claude-4.6-sonnet"``.
    name: str = "abstract"

    @abstractmethod
    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Send ``prompt`` to the model and return the response.

        Implementations should:
        - measure wall-clock latency
        - capture token counts and dollar cost if the provider reports them
        - catch transport/auth errors and surface them via ``error``,
          not raise — the runner records failed attempts as data, not crashes
        """


# ---------------------------------------------------------------- registry

_ModelFactory = Callable[..., ModelAdapter]
_REGISTRY: dict[str, _ModelFactory] = {}


def register_model(name: str) -> Callable[[_ModelFactory], _ModelFactory]:
    """Decorator: register a factory under a provider:model id."""

    def deco(factory: _ModelFactory) -> _ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def make_model(name: str, **kwargs: Any) -> ModelAdapter:
    """Build a model adapter by provider:model id.

    Extra ``kwargs`` are passed through to the factory (e.g. ``api_key``).
    Raises ``KeyError`` if the name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(_REGISTRY)} "
            f"(adapters self-register on import — try `import ebit_gym.models` first)"
        )
    return _REGISTRY[name](**kwargs)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)
