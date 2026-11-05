"""
Tests for brain/resilience/ (Epic 5 / Story 5.2, Task 8).

Run with: pytest tests/test_resilience.py -v
"""

import asyncio

import pytest

from brain.resilience.retry import RetryPolicy, retry_async
from brain.resilience.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
from brain.resilience.provider_resilience import ResilientProvider, ProviderCallError
from brain.resilience.tool_resilience import run_tool_with_retry, ToolCallError


# --- RetryPolicy / retry_async (8.1, 8.5: retry-then-succeed, retry-exhausted) --

def test_retry_succeeds_on_second_attempt():
    async def run():
        calls = {"count": 0}

        async def flaky():
            calls["count"] += 1
            if calls["count"] < 2:
                raise ConnectionError("transient")
            return "ok"

        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0)
        result = await retry_async(flaky, policy=policy)
        assert result == "ok"
        assert calls["count"] == 2

    asyncio.run(run())


def test_retry_exhausted_raises_last_exception():
    async def run():
        calls = {"count": 0}

        async def always_fails():
            calls["count"] += 1
            raise ConnectionError(f"attempt {calls['count']}")

        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0)
        with pytest.raises(ConnectionError, match="attempt 3"):
            await retry_async(always_fails, policy=policy)
        assert calls["count"] == 3

    asyncio.run(run())


def test_retry_does_not_retry_non_retryable_exceptions():
    async def run():
        calls = {"count": 0}

        async def fails_with_value_error():
            calls["count"] += 1
            raise ValueError("not retryable")

        policy = RetryPolicy(max_attempts=3, base_delay=0.01, retryable_exceptions=(ConnectionError,))
        with pytest.raises(ValueError):
            await retry_async(fails_with_value_error, policy=policy)
        assert calls["count"] == 1  # no retries for an exception type not in the policy

    asyncio.run(run())


def test_retry_backoff_delay_increases_and_is_capped():
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=3.0, exponential_base=2.0, jitter=0)
    delays = [policy.delay_for_attempt(n) for n in range(1, 5)]
    assert delays == [1.0, 2.0, 3.0, 3.0]  # 1, 2, 4->capped at 3, 8->capped at 3


# --- CircuitBreaker (8.2, 8.5: circuit-open scenario) ----------------------

def test_circuit_breaker_opens_after_threshold_failures():
    async def run():
        breaker = CircuitBreaker("test-provider", failure_threshold=3, recovery_timeout=60)

        async def always_fails():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(always_fails)

        assert breaker.state == CircuitState.OPEN

        # Further calls are rejected immediately, without invoking the function.
        call_count = {"n": 0}

        async def should_not_run():
            call_count["n"] += 1

        with pytest.raises(CircuitOpenError):
            await breaker.call(should_not_run)
        assert call_count["n"] == 0

    asyncio.run(run())


def test_circuit_breaker_resets_failure_count_on_success():
    async def run():
        breaker = CircuitBreaker("test-provider-2", failure_threshold=3, recovery_timeout=60)

        async def fails():
            raise RuntimeError("boom")

        async def succeeds():
            return "ok"

        with pytest.raises(RuntimeError):
            await breaker.call(fails)
        with pytest.raises(RuntimeError):
            await breaker.call(fails)
        # A success before hitting the threshold resets the streak.
        await breaker.call(succeeds)
        assert breaker.state == CircuitState.CLOSED

        with pytest.raises(RuntimeError):
            await breaker.call(fails)
        with pytest.raises(RuntimeError):
            await breaker.call(fails)
        # Still closed - only 2 consecutive failures since the reset, below threshold of 3.
        assert breaker.state == CircuitState.CLOSED

    asyncio.run(run())


def test_circuit_breaker_half_opens_after_recovery_timeout_and_closes_on_success():
    async def run():
        breaker = CircuitBreaker("test-provider-3", failure_threshold=1, recovery_timeout=0.05)

        async def fails():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await breaker.call(fails)
        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.1)  # let the recovery timeout elapse
        assert breaker.state == CircuitState.HALF_OPEN

        async def succeeds():
            return "recovered"

        result = await breaker.call(succeeds)
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED

    asyncio.run(run())


def test_circuit_breaker_reopens_if_half_open_trial_fails():
    async def run():
        breaker = CircuitBreaker("test-provider-4", failure_threshold=1, recovery_timeout=0.05)

        async def fails():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await breaker.call(fails)
        await asyncio.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            await breaker.call(fails)  # trial call also fails
        assert breaker.state == CircuitState.OPEN

    asyncio.run(run())


def test_circuit_breaker_get_status_shape():
    breaker = CircuitBreaker("status-test", failure_threshold=5, recovery_timeout=30)
    status = breaker.get_status()
    assert status == {
        "name": "status-test",
        "state": "closed",
        "failure_count": 0,
        "failure_threshold": 5,
        "recovery_timeout_seconds": 30,
    }


# --- ResilientProvider (8.3, 8.5: end-to-end through both layers) ---------

class _FlakyInnerProvider:
    """
    A provider double matching this codebase's actual error convention:
    generate_response() returns {"success": False, ...} on failure rather
    than raising (see brain/providers/openai.py).
    """

    def __init__(self, fail_times: int = 0, always_fail: bool = False):
        self.name = "flaky"
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.call_count = 0
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self):
        self.initialize_calls += 1
        return True

    async def close(self):
        self.close_calls += 1

    async def health_check(self):
        return {"healthy": True, "provider": self.name}

    async def generate_response(self, messages, **kwargs):
        self.call_count += 1
        if self.always_fail or self.call_count <= self.fail_times:
            return {"content": "", "success": False, "metadata": {"error": f"failure #{self.call_count}"}}
        return {"content": "recovered response", "success": True, "metadata": {}}


def test_resilient_provider_retries_returned_failure_dict_and_succeeds():
    """
    The key adaptation this module makes: a provider adapter returning
    {"success": False, ...} (not raising) must still trigger a retry.
    """
    async def run():
        inner = _FlakyInnerProvider(fail_times=1)
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0,
                              retryable_exceptions=(ProviderCallError,))
        wrapped = ResilientProvider(inner, retry_policy=policy, circuit_breaker=CircuitBreaker("flaky-test-1", failure_threshold=10, recovery_timeout=60))

        response = await wrapped.generate_response(messages=[])
        assert response["success"] is True
        assert response["content"] == "recovered response"
        assert inner.call_count == 2  # first failed, second succeeded

    asyncio.run(run())


def test_resilient_provider_returns_error_dict_after_retries_exhausted():
    async def run():
        inner = _FlakyInnerProvider(always_fail=True)
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0,
                              retryable_exceptions=(ProviderCallError,))
        breaker = CircuitBreaker("flaky-test-2", failure_threshold=10, recovery_timeout=60)
        wrapped = ResilientProvider(inner, retry_policy=policy, circuit_breaker=breaker)

        response = await wrapped.generate_response(messages=[])
        assert response["success"] is False
        assert "error" in response["metadata"]
        assert inner.call_count == 3  # all 3 attempts used

    asyncio.run(run())


def test_resilient_provider_opens_circuit_and_fails_fast_across_calls():
    """
    Failures accumulate on the *shared* circuit breaker across separate
    generate_response() calls (not just within one call's retries), and once
    open, subsequent calls fail fast without ever reaching the inner provider.
    """
    async def run():
        inner = _FlakyInnerProvider(always_fail=True)
        # No retries within a single call, so each call contributes exactly
        # one failure to the breaker's count.
        policy = RetryPolicy(max_attempts=1, base_delay=0.01, retryable_exceptions=(ProviderCallError,))
        breaker = CircuitBreaker("flaky-test-3", failure_threshold=2, recovery_timeout=60)
        wrapped = ResilientProvider(inner, retry_policy=policy, circuit_breaker=breaker)

        r1 = await wrapped.generate_response(messages=[])
        assert r1["success"] is False
        r2 = await wrapped.generate_response(messages=[])
        assert r2["success"] is False
        assert breaker.state == CircuitState.OPEN

        calls_before = inner.call_count
        r3 = await wrapped.generate_response(messages=[])
        assert r3["success"] is False
        assert r3["metadata"].get("circuit_breaker") == "open"
        assert inner.call_count == calls_before  # inner provider was never called - fast fail

    asyncio.run(run())


def test_resilient_provider_health_check_includes_circuit_breaker_status():
    async def run():
        inner = _FlakyInnerProvider()
        breaker = CircuitBreaker("flaky-test-4", failure_threshold=5, recovery_timeout=30)
        wrapped = ResilientProvider(inner, circuit_breaker=breaker)

        health = await wrapped.health_check()
        assert health["healthy"] is True
        assert health["circuit_breaker"]["name"] == "flaky-test-4"
        assert health["circuit_breaker"]["state"] == "closed"

    asyncio.run(run())


def test_resilient_provider_passes_through_initialize_and_close():
    async def run():
        inner = _FlakyInnerProvider()
        wrapped = ResilientProvider(inner)

        assert await wrapped.initialize() is True
        assert inner.initialize_calls == 1

        await wrapped.close()
        assert inner.close_calls == 1

    asyncio.run(run())


def test_create_provider_wraps_result_in_resilient_provider(monkeypatch):
    """Integration point: the factory (brain/providers/__init__.py) applies
    this wrapping automatically, so WorkflowEngine gets it for free."""
    from brain.providers import create_provider
    import brain.providers.openai as openai_module

    class DummyOpenAI:
        def __init__(self, config):
            self.name = "openai"

    monkeypatch.setattr(openai_module, "OpenAIProvider", DummyOpenAI)
    monkeypatch.setattr(openai_module, "create_openai_provider", lambda config: DummyOpenAI(config))

    import brain.providers as providers_pkg
    if "OpenAIProvider" not in providers_pkg.__all__:
        pytest.skip("OpenAI provider not available in this environment")

    provider = create_provider("openai", {})
    assert isinstance(provider, ResilientProvider)
    assert provider.name == "openai"


# --- run_tool_with_retry (8.1: retry also covers tool failures, not just --
# --- providers - found missing during the PR7 pre-audit) ------------------

class _FlakyTool:
    """
    A tool double matching this codebase's actual convention:
    execute() returns ToolResult(success=False, ...) on failure rather than
    raising (see brain/tools/tavily_search.py).
    """

    def __init__(self, fail_times: int = 0, always_fail: bool = False, hang_seconds: float = 0.0):
        self.name = "flaky_tool"
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.hang_seconds = hang_seconds
        self.call_count = 0
        self.close_calls = 0

    async def initialize(self) -> bool:
        return True

    async def close(self) -> None:
        self.close_calls += 1

    def get_parameters(self) -> dict:
        return {}

    async def execute(self, **kwargs):
        from brain.tools.base_tool import ToolResult
        self.call_count += 1
        if self.hang_seconds:
            await asyncio.sleep(self.hang_seconds)
        if self.always_fail or self.call_count <= self.fail_times:
            return ToolResult(tool_name=self.name, success=False, content=None, metadata={}, error=f"failure #{self.call_count}")
        return ToolResult(tool_name=self.name, success=True, content="tool ok", metadata={})


def test_run_tool_with_retry_succeeds_after_transient_failure():
    async def run():
        tool = _FlakyTool(fail_times=1)
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0,
                              retryable_exceptions=(ToolCallError,))
        result = await run_tool_with_retry(tool, timeout_seconds=5, retry_policy=policy)
        assert result.success is True
        assert result.content == "tool ok"
        assert tool.call_count == 2
        # managed_tool (brain/lifecycle.py) still closes the tool after every
        # attempt, including the failed one.
        assert tool.close_calls == 2

    asyncio.run(run())


def test_run_tool_with_retry_returns_last_failure_after_exhausted():
    async def run():
        tool = _FlakyTool(always_fail=True)
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0,
                              retryable_exceptions=(ToolCallError,))
        result = await run_tool_with_retry(tool, timeout_seconds=5, retry_policy=policy)
        assert result.success is False
        assert "failure #3" in result.error
        assert tool.call_count == 3

    asyncio.run(run())


def test_run_tool_with_retry_returns_failure_result_on_timeout_exhausted():
    async def run():
        tool = _FlakyTool(hang_seconds=1.0)
        policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.02, jitter=0,
                              retryable_exceptions=(ToolCallError, TimeoutError))
        result = await run_tool_with_retry(tool, timeout_seconds=0.05, retry_policy=policy)
        assert result.success is False
        assert result.error  # some timeout-related message
        assert tool.call_count == 2  # retried once after the first timeout
        assert tool.close_calls == 2  # closed on every attempt despite timing out

    asyncio.run(run())


def test_execute_tools_node_retries_transient_tool_failures(monkeypatch):
    """Integration point: WorkflowEngine._execute_one_tool goes through
    run_tool_with_retry automatically."""
    async def run():
        import brain.tools as tools_pkg
        from brain.tools.tool_registry import get_tool_registry
        from brain.cortex.workflow_engine import WorkflowEngine
        from brain.specs import BrainRequest, UserContext, ToolSpec

        flaky = _FlakyTool(fail_times=1)
        monkeypatch.setattr(tools_pkg, "create_tool", lambda tool_type, config: flaky)
        monkeypatch.setattr(get_tool_registry(), "check_tool_available", lambda name: True)

        engine = WorkflowEngine()
        state = {
            "request": BrainRequest(
                id="t-tool-retry", application="general", message="hi",
                user_context=UserContext(user_id="u", session_id="s", is_authenticated=False),
            ),
            "specs": type("Specs", (), {"tools": [ToolSpec(name="flaky_tool", enabled=True)]})(),
            "execution_steps": [], "errors": [], "tool_results": [],
        }

        result_state = await engine._execute_tools_node(state)
        assert result_state["tool_results"][0]["status"] == "executed"
        assert flaky.call_count == 2  # first attempt failed, retry succeeded

    asyncio.run(run())
