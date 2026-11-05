"""LLM_PROVIDER / LLM_MODEL drive which provider the agents request."""
import asyncio
import pytest
from config import settings
from plugins.echo_agent import EchoAgent
from brain.specs import AgentConfig, BrainRequest


def _spec_provider(monkeypatch, provider, model=None):
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "llm_model", model)
    return settings.default_llm()


def test_default_is_openai(monkeypatch):
    assert _spec_provider(monkeypatch, "openai") == ("openai", settings.openai_model)


def test_openrouter_uses_registry_default(monkeypatch):
    assert _spec_provider(monkeypatch, "openrouter") == ("openrouter", "openai/gpt-4o-mini")


def test_explicit_model_wins(monkeypatch):
    assert _spec_provider(monkeypatch, "openrouter", "anthropic/claude-3.5-sonnet")[1] == "anthropic/claude-3.5-sonnet"


def test_startup_flags_missing_key_and_bad_model(monkeypatch):
    _spec_provider(monkeypatch, "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    assert any("OPENROUTER_API_KEY" in p for p in settings.validate_required_for_startup())
    monkeypatch.setattr(settings, "openrouter_api_key", "k")
    assert settings.validate_required_for_startup() == []
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")  # openai model on openrouter
    assert any("invalid" in p for p in settings.validate_required_for_startup())


def test_agent_spec_follows_setting(monkeypatch):
    _spec_provider(monkeypatch, "openrouter")
    agent = EchoAgent(AgentConfig(agent_id="echo"))
    spec = asyncio.run(agent.create_execution_specs(None))
    assert spec.provider.provider_type == "openrouter"
