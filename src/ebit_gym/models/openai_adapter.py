"""OpenAI chat-completion adapter.

Requires ``pip install ebit-gym[models]``. API key from ``OPENAI_API_KEY``
unless passed explicitly. Supports any chat-completion model OpenAI exposes;
register one factory per model id at module load time.
"""
from __future__ import annotations

import os
import time

from ebit_gym.core.model import ModelAdapter, ModelResponse, register_model
from ebit_gym.models._pricing import estimate_usd


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAIAdapter requires the [models] extra: pip install ebit-gym[models]"
            ) from e

        self.model = model
        self.name = name or f"openai:{model}"
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:  # broad: network, auth, rate-limit, etc.
            return ModelResponse(
                raw_response="",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        in_toks = getattr(usage, "prompt_tokens", None) if usage else None
        out_toks = getattr(usage, "completion_tokens", None) if usage else None
        usd = estimate_usd(self.name, in_toks or 0, out_toks or 0) if in_toks else None

        return ModelResponse(
            raw_response=text,
            input_tokens=in_toks,
            output_tokens=out_toks,
            usd=usd,
            latency_ms=latency_ms,
            metadata={"finish_reason": getattr(choice, "finish_reason", None)},
        )


# Register the most common chat-completion model ids. Callers can also
# build OpenAIAdapter directly for niche models.
for _model in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "o1", "o1-mini"):

    @register_model(f"openai:{_model}")
    def _factory(model: str = _model, **kwargs: object) -> OpenAIAdapter:  # noqa: B008
        return OpenAIAdapter(model, **kwargs)  # type: ignore[arg-type]
