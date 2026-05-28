"""Reference model adapter implementations.

Importing this module registers all built-in adapters so ``make_model``
can find them by provider:model id.
"""

from ebit_gym.models.anthropic_adapter import AnthropicAdapter
from ebit_gym.models.mock import MockAdapter
from ebit_gym.models.openai_adapter import OpenAIAdapter
from ebit_gym.models.upstage import UpstageAdapter

__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
    "UpstageAdapter",
    "MockAdapter",
]
