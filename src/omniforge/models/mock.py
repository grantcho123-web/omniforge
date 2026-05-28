"""Mock model adapter for tests, demos, and offline development.

Either return canned strings in sequence, or call a custom callable for
fully programmatic responses. Never hits a network. Records every call
for assertion in tests.
"""
from __future__ import annotations

from collections.abc import Callable

from omniforge.core.model import ModelAdapter, ModelResponse, register_model


class MockAdapter(ModelAdapter):
    def __init__(
        self,
        responses: list[str] | None = None,
        respond_with: Callable[[str], str] | None = None,
        name: str = "mock:default",
        latency_ms: int = 0,
    ) -> None:
        if responses is None and respond_with is None:
            raise ValueError("MockAdapter needs either 'responses' or 'respond_with'")
        self.name = name
        self._responses = list(responses or [])
        self._respond_with = respond_with
        self._latency_ms = latency_ms
        self.calls: list[dict] = []

    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._respond_with is not None:
            text = self._respond_with(prompt)
        else:
            if not self._responses:
                return ModelResponse(raw_response="", error="MockAdapter exhausted")
            text = self._responses.pop(0)
        return ModelResponse(
            raw_response=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            usd=0.0,
            latency_ms=self._latency_ms,
        )


@register_model("mock:default")
def _factory(**kwargs: object) -> MockAdapter:
    # Default mock always returns "ok"; tests usually construct MockAdapter directly.
    return MockAdapter(responses=["ok"], **kwargs)  # type: ignore[arg-type]
