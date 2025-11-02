"""
Brain Providers - AI provider implementations for ThinkLife Brain
"""

# Local provider removed

__all__ = []  # All providers removed except OpenAI

# Optional providers
try:
    from .openai import OpenAIProvider
    __all__.append("OpenAIProvider")
except ImportError:
    pass

try:
    from .anthropic import AnthropicProvider
    __all__.append("AnthropicProvider")
except ImportError:
    pass 