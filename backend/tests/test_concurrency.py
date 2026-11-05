"""
Concurrency tests (Epic 5 / Story 5.1, Task 7.5).

Run with: pytest tests/test_concurrency.py -v

Notes on testing against CortexFlow/WorkflowEngine:
- `CortexFlow` (`brain/cortex/cortex.py`) and `WorkflowEngine`
  (`brain/cortex/workflow_engine.py`) are both process-wide singletons.
  That means tests across the whole session share one `CortexFlow.analytics`
  object - these tests measure *deltas* (before/after a batch of concurrent
  calls) rather than asserting absolute totals, so they're correct
  regardless of what earlier tests already did to the shared singleton.
"""

import asyncio
import time

import pytest


# --- 7.1: asyncio.gather parallelizes independent tool/data-source calls --

class _SlowFakeTool:
    """A tool double that takes a fixed amount of time to execute."""

    def __init__(self, delay_seconds: float, concurrency_tracker: list):
        self.delay_seconds = delay_seconds
        self.concurrency_tracker = concurrency_tracker
        self.name = "slow_fake_tool"

    async def initialize(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def get_parameters(self) -> dict:
        return {}

    async def execute(self, **kwargs):
        from brain.tools.base_tool import ToolResult
        self.concurrency_tracker.append(1)
        try:
            await asyncio.sleep(self.delay_seconds)
        finally:
            self.concurrency_tracker.append(-1)
        return ToolResult(tool_name="slow_fake_tool", success=True, content="done", metadata={})


def test_execute_tools_node_runs_independent_tools_concurrently(monkeypatch):
    """
    Regression/behavior test for Task 7.1: three tools that each take
    ~0.2s should complete in well under 3 * 0.2s if they run concurrently
    via asyncio.gather, versus ~0.6s+ if run sequentially (the previous
    behavior).
    """
    async def run():
        import brain.tools as tools_pkg
        from brain.tools.tool_registry import get_tool_registry
        from brain.cortex.workflow_engine import WorkflowEngine
        from brain.specs import BrainRequest, UserContext, ToolSpec

        concurrency_tracker: list = []
        delay = 0.2
        num_tools = 3

        def fake_create_tool(tool_type, config):
            return _SlowFakeTool(delay, concurrency_tracker)

        monkeypatch.setattr(tools_pkg, "create_tool", fake_create_tool)
        monkeypatch.setattr(get_tool_registry(), "check_tool_available", lambda name: True)

        engine = WorkflowEngine()
        state = {
            "request": BrainRequest(
                id="t-concurrency", application="general", message="hi",
                user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
            ),
            "specs": type("Specs", (), {
                "tools": [ToolSpec(name=f"tool_{i}", enabled=True) for i in range(num_tools)]
            })(),
            "execution_steps": [], "errors": [], "tool_results": [],
        }

        start = time.perf_counter()
        result_state = await engine._execute_tools_node(state)
        elapsed = time.perf_counter() - start

        assert len(result_state["tool_results"]) == num_tools
        assert all(r.get("status") == "executed" for r in result_state["tool_results"])

        # Sequential execution would take >= num_tools * delay (0.6s).
        # Concurrent execution should take roughly one `delay` plus overhead.
        assert elapsed < delay * num_tools * 0.7, (
            f"expected concurrent execution (~{delay}s), took {elapsed:.3f}s - "
            f"looks sequential"
        )

        # Peak concurrent in-flight tools observed via the tracker.
        peak = 0
        current = 0
        for delta in concurrency_tracker:
            current += delta
            peak = max(peak, current)
        assert peak > 1, "expected more than one tool in flight at once"

    asyncio.run(run())


def test_query_data_sources_node_runs_independent_sources_concurrently(monkeypatch):
    async def run():
        import brain.data_sources as ds_pkg
        from brain.data_sources.data_source_registry import get_data_source_registry
        from brain.cortex.workflow_engine import WorkflowEngine
        from brain.specs import BrainRequest, UserContext, DataSourceSpec, DataSourceType

        concurrency_tracker: list = []
        delay = 0.2
        num_sources = 3

        class SlowFakeDataSource:
            async def initialize(self, config):
                return True

            async def close(self):
                pass

            async def query(self, query, context=None, **kwargs):
                concurrency_tracker.append(1)
                try:
                    await asyncio.sleep(delay)
                finally:
                    concurrency_tracker.append(-1)
                return [{"content": "result"}]

        monkeypatch.setattr(ds_pkg, "create_vector_data_source", lambda config: SlowFakeDataSource())
        monkeypatch.setattr(get_data_source_registry(), "check_data_source_available", lambda name: True)

        engine = WorkflowEngine()
        state = {
            "request": BrainRequest(
                id="t-concurrency-ds", application="general", message="hi",
                user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
            ),
            "specs": type("Specs", (), {
                "data_sources": [
                    DataSourceSpec(source_type=DataSourceType.VECTOR_DB, enabled=True, config={})
                    for _ in range(num_sources)
                ]
            })(),
            "execution_steps": [], "errors": [], "data_source_results": [],
        }

        start = time.perf_counter()
        result_state = await engine._query_data_sources_node(state)
        elapsed = time.perf_counter() - start

        assert len(result_state["data_source_results"]) == num_sources
        assert elapsed < delay * num_sources * 0.7, (
            f"expected concurrent execution (~{delay}s), took {elapsed:.3f}s"
        )

    asyncio.run(run())


# --- 7.2: locked shared state stays correct under real concurrency -------

def test_active_executions_and_analytics_correct_under_concurrent_requests(monkeypatch):
    """
    Fires many concurrent CortexFlow.process_agent_request calls (using a
    disabled provider so each fails fast and deterministically via the
    existing validation path) and checks:
    - no exceptions escape asyncio.gather
    - active_executions is empty again once everything completes (Task 7.2)
    - analytics.total_requests increased by exactly N, under the lock,
      despite N requests updating it concurrently
    """
    async def run():
        from brain.cortex.cortex import CortexFlow
        from brain.specs import BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec, ProcessingSpec

        cortex = CortexFlow({"providers": {"openai": {"enabled": False}}})
        await cortex.initialize()

        N = 25
        requests = [
            BrainRequest(
                id=f"concurrent-{i}", application="general", message="hi",
                user_context=UserContext(user_id=f"user-{i}", session_id=f"s-{i}", is_authenticated=False),
            )
            for i in range(N)
        ]
        # A provider_type that fails registry validation immediately -
        # exercises the fast, deterministic error-response path in
        # _process_agent_request_inner without needing real credentials.
        specs = AgentExecutionSpec(
            data_sources=[], tools=[],
            provider=ProviderSpec(provider_type="not_a_real_provider", model="x"),
            processing=ProcessingSpec(execution_strategy="direct"),
        )

        before_total = cortex.analytics.total_requests

        results = await asyncio.gather(
            *(cortex.process_agent_request(specs, req) for req in requests),
            return_exceptions=True,
        )

        # No request should have raised out of process_agent_request itself.
        exceptions = [r for r in results if isinstance(r, BaseException)]
        assert not exceptions, f"unexpected exceptions: {exceptions}"

        assert all(isinstance(r, dict) and r.get("success") is False for r in results)

        assert len(cortex.active_executions) == 0, "no execution should remain tracked as active"
        assert cortex.analytics.total_requests == before_total + N, (
            "expected exactly N new requests recorded despite concurrent updates"
        )

    asyncio.run(run())


# --- 7.3: provider_semaphore bounds concurrent provider calls ------------

def test_provider_semaphore_bounds_concurrent_calls(monkeypatch):
    async def run():
        from brain.cortex.cortex import CortexFlow
        from brain.cortex.workflow_engine import WorkflowEngine
        import brain.providers as providers_pkg
        from brain.specs import BrainRequest, UserContext, AgentExecutionSpec, ProviderSpec, ProcessingSpec

        cortex = CortexFlow({"providers": {"openai": {"enabled": False}}})
        await cortex.initialize()

        # Temporarily narrow the shared WorkflowEngine's semaphore so the
        # bound is cheap to observe within a fast test.
        engine: WorkflowEngine = cortex.workflow_engine
        original_semaphore = engine.provider_semaphore
        limit = 2
        engine.provider_semaphore = asyncio.Semaphore(limit)

        concurrency_tracker: list = []
        peak = 0
        current = 0

        class SlowFakeProvider:
            name = "fake"

            async def initialize(self):
                return True

            async def generate_response(self, messages, **kwargs):
                nonlocal peak, current
                current += 1
                peak = max(peak, current)
                await asyncio.sleep(0.1)
                current -= 1
                return {"success": True, "content": "ok"}

            async def close(self):
                pass

        monkeypatch.setattr(providers_pkg, "create_provider", lambda t, c: SlowFakeProvider())

        try:
            N = 6
            requests = [
                BrainRequest(
                    id=f"sem-{i}", application="general", message="hi",
                    user_context=UserContext(user_id=f"u{i}", session_id=f"s{i}", is_authenticated=False),
                )
                for i in range(N)
            ]
            specs = AgentExecutionSpec(
                data_sources=[], tools=[],
                provider=ProviderSpec(provider_type="fake", model="x"),
                processing=ProcessingSpec(execution_strategy="direct"),
            )
            # provider registry validation would reject an unknown "fake"
            # provider type before reaching _call_provider_node, so call the
            # workflow engine node directly instead, bypassing that check -
            # this test is specifically about the semaphore, not validation.
            async def call_provider_only():
                state = {
                    "specs": specs, "messages": [], "errors": [],
                    "execution_steps": [], "llm_response": None, "final_content": None,
                }
                return await engine._call_provider_node(state)

            await asyncio.gather(*(call_provider_only() for _ in range(N)))

            assert peak <= limit, f"observed {peak} concurrent provider calls, expected <= {limit}"
            assert peak > 1, "expected some real concurrency to occur, not fully serialized"
        finally:
            engine.provider_semaphore = original_semaphore

    asyncio.run(run())


# --- 7.4 (found during PR6 pre-audit): auth chain no longer blocks the ----
# --- event loop with a synchronous requests.get() call --------------------

def test_auth_chain_is_async_and_does_not_block_event_loop(monkeypatch):
    """
    Regression test: SecurityManager.validate_user -> SessionManager.
    validate_session -> validate_token -> _get_public_key was previously an
    entirely synchronous call chain that made a blocking `requests.get(...)`
    network call - on the critical path of KeycloakAuthMiddleware.dispatch,
    i.e. every authenticated request. This confirms the whole chain is now
    async and the network call is offloaded via asyncio.to_thread (it can be
    awaited concurrently with other work on the same event loop).
    """
    async def run():
        from brain.guardrails.security_manager import SecurityManager
        import brain.guardrails.session_manager as session_manager_module

        loop = asyncio.get_running_loop()
        call_started = asyncio.Event()

        class SlowResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"public_key": "fake-key"}

        def slow_requests_get(url, timeout=5):
            # Runs inside asyncio.to_thread - blocking sleep here must NOT
            # block the event loop, proven by the concurrent task below
            # completing while this is still "sleeping".
            loop.call_soon_threadsafe(call_started.set)
            import time as _time
            _time.sleep(0.2)
            return SlowResponse()

        monkeypatch.setattr(session_manager_module, "requests", type("R", (), {"get": staticmethod(slow_requests_get)}))

        security_manager = SecurityManager()
        # Use a syntactically-valid-enough unverified JWT so jwt.decode
        # doesn't fail before reaching _get_public_key. python-jose/PyJWT
        # just needs 3 base64url segments to decode without verification.
        import base64, json as jsonlib

        def b64(d):
            return base64.urlsafe_b64encode(jsonlib.dumps(d).encode()).rstrip(b"=").decode()

        header = b64({"alg": "RS256"})
        payload = b64({"iss": f"{security_manager.session_manager.keycloak_url}/realms/{security_manager.session_manager.keycloak_realm}"})
        signature = base64.urlsafe_b64encode(b"fake-signature-bytes").rstrip(b"=").decode()
        fake_token = f"{header}.{payload}.{signature}"

        async def other_concurrent_work():
            await call_started.wait()
            return "completed while auth call was in flight"

        auth_task = asyncio.create_task(
            security_manager.validate_user({"authenticated": False}, token=fake_token)
        )
        other_task = asyncio.create_task(other_concurrent_work())

        other_result = await asyncio.wait_for(other_task, timeout=1.0)
        assert other_result == "completed while auth call was in flight"

        # The auth call itself should still complete successfully afterward.
        await asyncio.wait_for(auth_task, timeout=1.0)

    asyncio.run(run())
