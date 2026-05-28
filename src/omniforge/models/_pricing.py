"""USD-per-million-token pricing for cost estimation.

Indicative only — providers change prices and many adapters omit costs
entirely. Update from official pricing pages periodically. For
contracts where cost reporting matters, prefer the provider's own
billing API over this table.
"""
from __future__ import annotations

# (input_per_mtok, output_per_mtok) in USD.
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI — circa mid-2026, approximate.
    "openai:gpt-4o": (5.0, 15.0),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4.1": (5.0, 15.0),
    "openai:o1": (15.0, 60.0),
    "openai:o1-mini": (3.0, 12.0),
    # Anthropic — IDs use dashes throughout: family before version, e.g. claude-haiku-4-5
    "anthropic:claude-haiku-4-5": (1.0, 5.0),
    "anthropic:claude-sonnet-4-5": (3.0, 15.0),
    "anthropic:claude-opus-4-5": (15.0, 75.0),
    # Upstage Solar
    "upstage:solar-pro": (0.50, 1.50),
    "upstage:solar-mini": (0.10, 0.30),
}


def estimate_usd(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return rough USD cost, or None if the model is unpriced in our table."""
    if model_name not in PRICING:
        return None
    in_rate, out_rate = PRICING[model_name]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
