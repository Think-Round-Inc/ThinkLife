"""
Tests for brain/lifecycle.py's context managers (Epic 5 / Story 5.2, Task 6).

Run with: pytest tests/test_lifecycle.py -v

Uses fake tool/provider doubles (not real Tavily/OpenAI clients) so these
tests are fast, deterministic, and need no API keys - they exercise the
guarantee itself ("close() always runs"), not any particular vendor SDK.
"""

import asyncio

import pytest

from brain.lifecycle import managed_provider, managed_tool, managed_data_source, run_tool_with_timeout
from brain.tools.base_tool import BaseTool, ToolResult


class FakeTool(BaseTool):
    """Configurable tool double for exercising managed_tool / run_tool_with_timeout."""

    def __init__(self, config=None, *, raise_on_execute=None, sleep_seconds=0.0):
        super().__init__(config)
        self.raise_on_execute = raise_on_execute
        self.sleep_seconds = sleep_seconds
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self) -> bool:
        self.initialize_calls += 1
        self._initialized = True
        return True

    async def close(self) -> None:
        self.close_calls += 1
        self._initialized = False

    async def execute(self, **kwargs) -> ToolResult:
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.raise_on_execute:
            raise self.raise_on_execute
        return ToolResult(tool_name="fake", success=True, content="ok", metadata={})


class FakeProvider:
    """Configurable provider double satisfying LLMProviderProtocol's shape."""

    def __init__(self, config=None, *, init_ok=True, raise_on_init=None, raise_on_generate=None):
        self.name = "fake"
        self.config = config or {}
        self.init_ok = init_ok
        self.raise_on_init = raise_on_init
        self.raise_on_generate = raise_on_generate
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self) -> bool:
        self.initialize_calls += 1
        if self.raise_on_init:
            raise self.raise_on_init
        return self.init_ok

    async def generate_response(self, messages, **kwargs):
        if self.raise_on_generate:
            raise self.raise_on_generate
        return {"success": True, "content": "hi"}

    async def health_check(self):
        return {"healthy": True}

    async def close(self) -> None:
        self.close_calls += 1


# --- managed_tool / run_tool_with_timeout ---------------------------------

def test_managed_tool_closes_on_success():
    async def run():
        tool = FakeTool()
        async with managed_tool(tool) as t:
            assert t is tool
            assert tool.initialize_calls == 1
            assert tool.close_calls == 0
        assert tool.close_calls == 1

    asyncio.run(run())


def test_managed_tool_closes_even_when_body_raises():
    async def run():
        tool = FakeTool()
        with pytest.raises(ValueError):
            async with managed_tool(tool):
                raise ValueError("boom inside the with-block")
        assert tool.close_calls == 1  # released despite the exception

    asyncio.run(run())


def test_run_tool_with_timeout_closes_on_execute_exception():
    async def run():
        tool = FakeTool(raise_on_execute=RuntimeError("tool blew up"))
        with pytest.raises(RuntimeError):
            await run_tool_with_timeout(tool, timeout_seconds=5)
        assert tool.close_calls == 1

    asyncio.run(run())


def test_run_tool_with_timeout_closes_on_actual_timeout():
    async def run():
        tool = FakeTool(sleep_seconds=1.0)
        with pytest.raises(asyncio.TimeoutError):
            await run_tool_with_timeout(tool, timeout_seconds=0.05)
        # The tool's close() must still have run despite the timeout cutting
        # execute() off mid-flight - this is the core Task 6 guarantee.
        assert tool.close_calls == 1

    asyncio.run(run())


def test_run_tool_with_timeout_returns_result_on_success():
    async def run():
        tool = FakeTool()
        result = await run_tool_with_timeout(tool, timeout_seconds=5)
        assert result.success is True
        assert result.content == "ok"
        assert tool.close_calls == 1

    asyncio.run(run())


# --- managed_provider ------------------------------------------------------

def test_managed_provider_closes_on_success(monkeypatch):
    async def run():
        fake = FakeProvider()

        def fake_create_provider(provider_type, config):
            return fake

        import brain.providers as providers_pkg
        monkeypatch.setattr(providers_pkg, "create_provider", fake_create_provider)

        async with managed_provider("fake", {}) as provider:
            assert provider is fake
            response = await provider.generate_response(messages=[])
            assert response["success"] is True
        assert fake.close_calls == 1

    asyncio.run(run())


def test_managed_provider_closes_when_initialize_returns_false(monkeypatch):
    """
    Regression test: the original hand-rolled pattern in
    WorkflowEngine._call_provider_node raised *before* its try/finally when
    initialize() returned False, so close() was never called in that case.
    managed_provider fixes that - close() must still run here.
    """
    async def run():
        fake = FakeProvider(init_ok=False)

        import brain.providers as providers_pkg
        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: fake)

        with pytest.raises(RuntimeError, match="Failed to initialize"):
            async with managed_provider("fake", {}):
                pass  # pragma: no cover - should never reach the body
        assert fake.close_calls == 1

    asyncio.run(run())


def test_managed_provider_closes_when_generate_response_raises(monkeypatch):
    async def run():
        fake = FakeProvider(raise_on_generate=RuntimeError("provider exploded"))

        import brain.providers as providers_pkg
        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: fake)

        with pytest.raises(RuntimeError, match="provider exploded"):
            async with managed_provider("fake", {}) as provider:
                await provider.generate_response(messages=[])
        assert fake.close_calls == 1

    asyncio.run(run())


def test_managed_provider_swallows_close_errors_without_masking_original(monkeypatch):
    """
    If close() itself raises, that shouldn't mask an earlier exception from
    the with-block, and shouldn't crash the caller either - it's logged and
    swallowed (see brain/lifecycle.py).
    """
    async def run():
        class BadCloseProvider(FakeProvider):
            async def close(self):
                self.close_calls += 1
                raise RuntimeError("close also blew up")

        fake = BadCloseProvider()
        import brain.providers as providers_pkg
        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: fake)

        with pytest.raises(ValueError, match="original error"):
            async with managed_provider("fake", {}):
                raise ValueError("original error")
        assert fake.close_calls == 1  # close() was attempted despite raising itself

    asyncio.run(run())


# --- managed_data_source ---------------------------------------------------

class FakeDataSource:
    """Configurable IDataSource double for exercising managed_data_source."""

    def __init__(self, *, init_ok=True, raise_on_query=None):
        self.init_ok = init_ok
        self.raise_on_query = raise_on_query
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self, config):
        self.initialize_calls += 1
        return self.init_ok

    async def query(self, query, context=None, **kwargs):
        if self.raise_on_query:
            raise self.raise_on_query
        return [{"content": "result"}]

    async def close(self):
        self.close_calls += 1


def test_managed_data_source_closes_on_success():
    async def run():
        source = FakeDataSource()
        async with managed_data_source(source, {}) as s:
            results = await s.query("hi")
            assert results == [{"content": "result"}]
        assert source.close_calls == 1

    asyncio.run(run())


def test_managed_data_source_closes_when_query_raises():
    """
    Regression test: WorkflowEngine._query_data_sources_node previously
    called `vector_source.close()` only *after* a successful `query()` call
    - if query() raised, close() was never reached, leaking the data source
    client. managed_data_source fixes that.
    """
    async def run():
        source = FakeDataSource(raise_on_query=RuntimeError("query blew up"))
        with pytest.raises(RuntimeError, match="query blew up"):
            async with managed_data_source(source, {}) as s:
                await s.query("hi")
        assert source.close_calls == 1

    asyncio.run(run())


def test_managed_data_source_closes_when_initialize_fails():
    async def run():
        source = FakeDataSource(init_ok=False)
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            async with managed_data_source(source, {}):
                pass  # pragma: no cover
        assert source.close_calls == 1

    asyncio.run(run())


def test_query_data_sources_node_closes_source_even_on_query_failure(monkeypatch):
    """
    Integration-level regression test hitting the actual
    WorkflowEngine._query_data_sources_node code path (not just the
    lifecycle helper in isolation).
    """
    async def run():
        from brain.cortex.cortex import CortexFlow
        from brain.specs import (
            BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec,
            ProcessingSpec, DataSourceSpec, DataSourceType,
        )
        import brain.data_sources as data_sources_pkg
        from brain.data_sources.data_source_registry import get_data_source_registry

        failing_source = FakeDataSource(raise_on_query=RuntimeError("boom"))
        # _query_data_sources_node does `from ..data_sources import
        # create_vector_data_source` as a *local* import inside the method,
        # so it resolves this attribute on brain.data_sources fresh on each
        # call - patching it there (not on the workflow_engine module) is
        # what actually takes effect.
        monkeypatch.setattr(data_sources_pkg, "create_vector_data_source", lambda config: failing_source)
        monkeypatch.setattr(get_data_source_registry(), "check_data_source_available", lambda name: True)

        cortex = CortexFlow({"providers": {"openai": {"enabled": False}}})
        await cortex.initialize()
        try:
            state = {
                "request": BrainRequest(
                    id="t1", application="general", message="hi",
                    user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
                ),
                "specs": AgentExecutionSpec(
                    data_sources=[DataSourceSpec(source_type=DataSourceType.VECTOR_DB, enabled=True)],
                    provider=ProviderSpec(provider_type="openai", model="gpt-4o-mini"),
                    tools=[], processing=ProcessingSpec(execution_strategy="direct"),
                ),
                "execution_steps": [], "errors": [], "tool_results": [], "data_source_results": [],
                "messages": [], "validation_attempts": 0, "max_validation_runs": 5,
            }

            result_state = await cortex.workflow_engine._query_data_sources_node(state)
            assert failing_source.close_calls == 1, "close() must run even though query() raised"
        finally:
            await cortex.shutdown()

    asyncio.run(run())


# --- Integration: WorkflowEngine._execute_tools_node ----------------------

def test_execute_tools_node_no_longer_a_placeholder():
    """
    Regression test: _execute_tools_node previously returned a hard-coded
    'Tool execution placeholder' string without instantiating any tool.
    It should now attempt a real tool call (and fail gracefully without
    credentials, rather than returning the canned placeholder).
    """
    async def run():
        from brain.cortex.cortex import CortexFlow
        from brain.specs import BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec, ProcessingSpec, ToolSpec

        cortex = CortexFlow({"providers": {"openai": {"enabled": False}}})
        await cortex.initialize()
        try:
            state = {
                "request": BrainRequest(
                    id="t1", application="general", message="latest AI news",
                    user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
                ),
                "specs": AgentExecutionSpec(
                    data_sources=[], provider=ProviderSpec(provider_type="openai", model="gpt-4o-mini"),
                    tools=[ToolSpec(name="tavily_search", enabled=True)],
                    processing=ProcessingSpec(execution_strategy="direct"),
                ),
                "execution_steps": [], "errors": [], "tool_results": [], "data_source_results": [],
                "messages": [], "validation_attempts": 0, "max_validation_runs": 5,
            }
            result_state = await cortex.workflow_engine._execute_tools_node(state)
            tool_results = result_state["tool_results"]
            assert len(tool_results) == 1
            assert tool_results[0]["tool"] == "tavily_search"
            # No API key/library in the test environment -> a real (not
            # placeholder) failure reason should be present.
            assert "placeholder" not in str(tool_results[0]).lower()
            assert "error" in tool_results[0]
        finally:
            await cortex.shutdown()

    asyncio.run(run())
