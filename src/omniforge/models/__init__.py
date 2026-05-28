"""Reference model adapter implementations.

Importing this module registers all built-in adapters so ``make_model``
can find them by provider:model id.
"""

from omniforge.models.anthropic_adapter import AnthropicAdapter
from omniforge.models.mock import MockAdapter
from omniforge.models.openai_adapter import OpenAIAdapter
from omniforge.models.upstage import UpstageAdapter

__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
    "UpstageAdapter",
    "MockAdapter",
]
