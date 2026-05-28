"""Tests for the model-adapter layer.

Live API calls are intentionally NOT tested here — those need keys, cost
money, and are flaky. Instead we test:
- The registry + factory contract
- The MockAdapter end-to-end (used by the runner tests)
- OpenAI / Anthropic adapter shape via monkey-patched SDK clients
- Cost estimation
- Upstage adapter routes through the OpenAI base_url override
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

# Importing this registers all built-in adapters.
import omniforge.models  # noqa: F401
from omniforge.core.model import make_model, registered_models
from omniforge.models._pricing import PRICING, estimate_usd
from omniforge.models.mock import MockAdapter

# ----------------------------------------------------------------- registry


def test_registry_contains_expected_models():
    names = registered_models()
    assert "anthropic:claude-4.6-sonnet" in names
    assert "openai:gpt-4o-mini" in names
    assert "upstage:solar-pro" in names
    assert "mock:default" in names


def test_make_model_rejects_unknown():
    with pytest.raises(KeyError, match="unknown model"):
        make_model("not-a-real-model")


# ----------------------------------------------------------------- pricing


def test_estimate_usd_known_model():
    in_rate, out_rate = PRICING["anthropic:claude-4.6-sonnet"]
    cost = estimate_usd("anthropic:claude-4.6-sonnet", 1_000_000, 1_000_000)
    assert cost == pytest.approx(in_rate + out_rate)


def test_estimate_usd_unknown_model_returns_none():
    assert estimate_usd("nonexistent:model", 100, 100) is None


# --------------------------------------------------------------------- mock


def test_mock_returns_canned_responses_in_order():
    m = MockAdapter(responses=["first", "second", "third"])
    assert m.call("hi").raw_response == "first"
    assert m.call("hi").raw_response == "second"
    assert m.call("hi").raw_response == "third"


def test_mock_exhausted_returns_error():
    m = MockAdapter(responses=["one"])
    m.call("x")
    r = m.call("x")
    assert r.error == "MockAdapter exhausted"


def test_mock_responds_with_callable():
    m = MockAdapter(respond_with=lambda p: f"echo:{p[:5]}")
    r = m.call("hello world")
    assert r.raw_response == "echo:hello"


def test_mock_records_calls_for_assertion():
    m = MockAdapter(responses=["x"])
    m.call("the prompt", system="be terse", temperature=0.5)
    assert m.calls[0]["prompt"] == "the prompt"
    assert m.calls[0]["system"] == "be terse"
    assert m.calls[0]["temperature"] == 0.5


def test_mock_requires_one_of_two_inputs():
    with pytest.raises(ValueError, match="needs either"):
        MockAdapter()


def test_mock_returns_token_counts():
    m = MockAdapter(responses=["one two three"])
    r = m.call("hi there")
    assert r.input_tokens == 2
    assert r.output_tokens == 3


# -------------------------------------------------------------- openai shape


def test_openai_adapter_call_shape(monkeypatch):
    """Verify the OpenAI adapter produces a well-shaped ModelResponse
    given a stubbed SDK client. No real network call."""
    from omniforge.models.openai_adapter import OpenAIAdapter

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="The answer is 42."),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=14),
    )

    a = OpenAIAdapter(model="gpt-4o-mini", api_key="dummy")
    monkeypatch.setattr(
        a._client.chat.completions, "create", lambda **_: fake_response
    )
    r = a.call("question?", system="be brief", max_tokens=64, temperature=0.1)

    assert r.raw_response == "The answer is 42."
    assert r.input_tokens == 120
    assert r.output_tokens == 14
    assert r.usd is not None and r.usd > 0
    assert r.latency_ms is not None and r.latency_ms >= 0
    assert r.metadata["finish_reason"] == "stop"


def test_openai_adapter_surfaces_errors_via_field(monkeypatch):
    from omniforge.models.openai_adapter import OpenAIAdapter

    def boom(**_):
        raise RuntimeError("network down")

    a = OpenAIAdapter(model="gpt-4o-mini", api_key="dummy")
    monkeypatch.setattr(a._client.chat.completions, "create", boom)
    r = a.call("q")
    assert r.raw_response == ""
    assert r.error is not None
    assert "network down" in r.error
    assert "RuntimeError" in r.error


# ------------------------------------------------------------ anthropic shape


def test_anthropic_adapter_call_shape(monkeypatch):
    from omniforge.models.anthropic_adapter import AnthropicAdapter

    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="42.")],
        usage=SimpleNamespace(input_tokens=100, output_tokens=3),
        stop_reason="end_turn",
    )

    a = AnthropicAdapter(model="claude-4.6-sonnet", api_key="dummy")
    monkeypatch.setattr(a._client.messages, "create", lambda **_: fake_response)
    r = a.call("q", system="be brief")

    assert r.raw_response == "42."
    assert r.input_tokens == 100
    assert r.output_tokens == 3
    assert r.usd is not None and r.usd > 0
    assert r.metadata["stop_reason"] == "end_turn"


def test_anthropic_adapter_concatenates_text_blocks(monkeypatch):
    """Anthropic can return multiple content blocks; we join the text ones."""
    from omniforge.models.anthropic_adapter import AnthropicAdapter

    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Part one. "),
            SimpleNamespace(type="text", text="Part two."),
            SimpleNamespace(type="tool_use", input={}),  # should be skipped
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    a = AnthropicAdapter(model="claude-4.6-sonnet", api_key="dummy")
    monkeypatch.setattr(a._client.messages, "create", lambda **_: fake_response)
    r = a.call("q")
    assert r.raw_response == "Part one. Part two."


# --------------------------------------------------------------------- upstage


def test_upstage_adapter_uses_upstage_base_url():
    """Upstage adapter must route through Upstage's API endpoint, not OpenAI's."""
    from omniforge.models.upstage import UPSTAGE_BASE_URL, UpstageAdapter

    a = UpstageAdapter(model="solar-pro", api_key="dummy")
    # The inner OpenAI client should have been configured with Upstage's base URL.
    assert str(a._inner._client.base_url).rstrip("/") == UPSTAGE_BASE_URL.rstrip("/")
    assert a.name == "upstage:solar-pro"
