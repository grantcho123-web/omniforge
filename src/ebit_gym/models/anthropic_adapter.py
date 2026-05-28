"""Anthropic Messages-API adapter.

Requires ``pip install ebit-gym[models]``. API key from ``ANTHROPIC_API_KEY``
unless passed explicitly.
"""
from __future__ import annotations

import os
import time

from ebit_gym.core.model import ModelAdapter, ModelResponse, register_model
from ebit_gym.models._pricing import estimate_usd


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        name: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "AnthropicAdapter requires the [models] extra: pip install ebit-gym[models]"
            ) from e

        self.model = model
        self.name = name or f"anthropic:{model}"
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:
            return ModelResponse(
                raw_response="",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        # Anthropic returns a list of content blocks; for simple text replies
        # the first block holds the answer.
        text_parts = [
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", "") == "text"
        ]
        text = "".join(text_parts)
        usage = resp.usage
        in_toks = getattr(usage, "input_tokens", None)
        out_toks = getattr(usage, "output_tokens", None)
        usd = estimate_usd(self.name, in_toks or 0, out_toks or 0) if in_toks else None

        return ModelResponse(
            raw_response=text,
            input_tokens=in_toks,
            output_tokens=out_toks,
            usd=usd,
            latency_ms=latency_ms,
            metadata={"stop_reason": getattr(resp, "stop_reason", None)},
        )


for _model in ("claude-4.6-sonnet", "claude-4.7-sonnet", "claude-4.5-haiku", "claude-opus-4-5"):

    @register_model(f"anthropic:{_model}")
    def _factory(model: str = _model, **kwargs: object) -> AnthropicAdapter:  # noqa: B008
        return AnthropicAdapter(model, **kwargs)  # type: ignore[arg-type]
