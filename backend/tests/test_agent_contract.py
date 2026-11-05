"""
Contract tests for Epic 5 / Story 5.2 Task 3 (plugin interface contracts) and
Task 4 (health check endpoints).

Run with: pytest tests/test_agent_contract.py -v

Uses `asyncio.run()` inside plain `def test_...()` functions (rather than
`pytest.mark.asyncio`) to match this project's existing sync-only test style
(see `tests/test_guardrails.py`) without adding a new pytest-asyncio
dependency.
"""

import asyncio

import pytest

from brain.specs import AgentConfig, AgentResponse, BrainRequest, IAgent, UserContext
from plugins.echo_agent import EchoAgent, create_echo_agent
from plugins.registry import PluginRegistry


def _make_request(message: str = "hello") -> BrainRequest:
    """Minimal valid BrainRequest for exercising an agent end to end."""
    return BrainRequest(
        id="test-request",
        application="general",
        message=message,
        user_context=UserContext(user_id="test-user", session_id="test-session", is_authenticated=False),
    )


# --- IAgent contract -----------------------------------------------------

def test_echo_agent_satisfies_iagent_interface():
    """EchoAgent is a real, independent second implementation of IAgent -
    the practical proof that the plugin contract works for more than one
    agent (not just ZoeAgent)."""
    agent = create_echo_agent(AgentConfig(agent_id="echo"))
    assert isinstance(agent, IAgent)


def test_echo_agent_full_lifecycle():
    async def run():
        agent = EchoAgent(AgentConfig(agent_id="echo"))

        assert await agent.initialize(agent.config) is True

        spec = await agent.create_execution_specs(_make_request())
        assert spec.provider is not None
        assert spec.provider.provider_type == "openai"

        response = await agent.process_request(_make_request("hello world"))
        assert isinstance(response, AgentResponse)
        assert response.success is True
        assert "hello world" in response.content

        health = await agent.health_check()
        assert health["initialized"] is True

        await agent.shutdown()
        health_after = await agent.health_check()
        assert health_after["initialized"] is False

    asyncio.run(run())


def test_echo_agent_metadata_matches_contract():
    agent = create_echo_agent(AgentConfig(agent_id="echo"))
    meta = agent.metadata
    assert meta.name
    assert meta.version
    assert isinstance(meta.supported_applications, list)


# --- Plugin registry -----------------------------------------------------

def test_plugin_registry_register_and_create():
    registry = PluginRegistry()
    config = AgentConfig(agent_id="echo")
    instance = create_echo_agent(config)

    registry.register("echo", create_echo_agent, instance.metadata)

    assert registry.is_registered("echo")
    assert "echo" in registry.list_plugins()

    created = registry.create("echo", config)
    assert isinstance(created, EchoAgent)

    metadata = registry.get_metadata("echo")
    assert metadata.name == "Echo"


def test_plugin_registry_unknown_plugin_raises():
    registry = PluginRegistry()
    with pytest.raises(KeyError):
        registry.create("does-not-exist", AgentConfig(agent_id="x"))


def test_plugin_registry_reregistration_overwrites_without_error():
    registry = PluginRegistry()
    config = AgentConfig(agent_id="echo")
    instance = create_echo_agent(config)

    registry.register("echo", create_echo_agent, instance.metadata)
    registry.register("echo", create_echo_agent, instance.metadata)  # should warn, not raise
    assert registry.list_plugins().count("echo") == 1


# --- CortexFlow health/analytics (regression test for a real bug) --------

def test_cortex_health_and_analytics_do_not_raise():
    """
    Regression test: `main.py`'s `/api/brain/health` and `/api/brain/analytics`
    endpoints call `CortexFlow.get_health_status()` / `get_analytics()`, but
    prior to this fix neither method existed anywhere in the codebase, so
    both endpoints raised an unhandled AttributeError (-> 500) on every call.
    """
    from brain.cortex.cortex import CortexFlow

    async def run():
        cortex = CortexFlow({"providers": {"openai": {"enabled": False}}})
        await cortex.initialize()
        try:
            health = await cortex.get_health_status()
            assert health["overall"] in ("healthy", "degraded", "unhealthy")
            assert "providers" in health
            assert "system" in health

            analytics = await cortex.get_analytics()
            assert "total_requests" in analytics
        finally:
            await cortex.shutdown()

    asyncio.run(run())
