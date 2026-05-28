"""Upstage Solar adapter.

Upstage exposes an OpenAI-compatible Chat Completions API at
``https://api.upstage.ai/v1``, so we just point the OpenAI client at
their base URL. This is the v0.2 Asian-model signal for the GTM —
demonstrates that ebit-gym ships with native support for the Korean
foundation-model stack from day one.

API key from ``UPSTAGE_API_KEY``.
"""
from __future__ import annotations

import os

from ebit_gym.core.model import ModelAdapter, ModelResponse, register_model
from ebit_gym.models.openai_adapter import OpenAIAdapter

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"


class UpstageAdapter(ModelAdapter):
    def __init__(
        self,
        model: str = "solar-pro",
        *,
        api_key: str | None = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.name = name or f"upstage:{model}"
        # Reuse the OpenAI adapter's chat-completions logic with Upstage's base URL.
        self._inner = OpenAIAdapter(
            model=model,
            api_key=api_key or os.getenv("UPSTAGE_API_KEY"),
            base_url=UPSTAGE_BASE_URL,
            name=self.name,
        )

    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        return self._inner.call(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature
        )


for _model in ("solar-pro", "solar-mini"):

    @register_model(f"upstage:{_model}")
    def _factory(model: str = _model, **kwargs: object) -> UpstageAdapter:  # noqa: B008
        return UpstageAdapter(model, **kwargs)  # type: ignore[arg-type]
