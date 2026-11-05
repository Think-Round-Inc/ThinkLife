"""
Tests for brain/providers/openrouter.py.

OpenRouter reuses OpenAIProvider's AsyncOpenAI client pointed at OpenRouter's
base_url, so these tests check the wiring (registry, factory, config
defaults) rather than re-testing generate_response/stream_response, which
are already covered for OpenAIProvider.

Run with: pytest tests/test_openrouter_provider.py -v
"""

import asyncio

from brain.providers.openrouter import OpenRouterProvider, OPENROUTER_BASE_URL
from brain.providers.provider_registry import get_provider_registry


def test_openrouter_in_provider_registry():
    registry = get_provider_registry()
    is_valid, errors, info = registry.check_provider_and_model("openrouter", "openai/gpt-4o-mini")
    assert is_valid, errors
    assert info["model"] == "openai/gpt-4o-mini"


def test_openrouter_rejects_unknown_model():
    registry = get_provider_registry()
    is_valid, errors, _ = registry.check_provider_and_model("openrouter", "not-a-real-model")
    assert not is_valid
    assert errors


def test_create_provider_wires_openrouter(monkeypatch):
    from brain.providers import create_provider
    from brain.resilience import ResilientProvider

    provider = create_provider("openrouter", {"model": "openai/gpt-4o-mini"})
    assert isinstance(provider, ResilientProvider)
    assert provider.name == "openrouter"


def test_initialize_defaults_base_url_and_api_key(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")

    provider = OpenRouterProvider({"model": "openai/gpt-4o-mini"})
    ok = asyncio.run(provider.initialize())

    assert ok
    assert provider.config["base_url"] == OPENROUTER_BASE_URL
    assert provider.config["api_key"] == "test-key"
    assert provider.client is not None


def test_explicit_config_overrides_defaults():
    provider = OpenRouterProvider({
        "model": "openai/gpt-4o-mini",
        "api_key": "explicit-key",
        "base_url": "https://custom.example/api/v1",
    })
    ok = asyncio.run(provider.initialize())

    assert ok
    assert provider.config["base_url"] == "https://custom.example/api/v1"
    assert provider.config["api_key"] == "explicit-key"


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
