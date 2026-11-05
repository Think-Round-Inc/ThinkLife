"""
Applies retry + circuit breaker at the provider-adapter boundary.

Per the Task 8 plan: wrap at `brain/providers/*.py`'s boundary, not inside
`WorkflowEngine`, keeping the two concerns (orchestration vs. resilience)
separate. `ResilientProvider` wraps a real provider instance (whatever
`create_provider()` returns) and is otherwise a drop-in replacement -
`WorkflowEngine._call_provider_node` and `managed_provider`
(`brain/lifecycle.py`) don't need to know it's there.

Important adaptation to how this codebase actually reports errors: provider
adapters (see `brain/providers/openai.py`) catch their own exceptions
internally and return `{"success": False, "content": "", "metadata":
{"error": ...}}` rather than raising. A retry/circuit-breaker layer that only
watches for raised exceptions would never see these failures. `ProviderCallError`
bridges that gap: `_call_and_raise_on_failure` raises it whenever the
wrapped call returns `success: False`, so both `retry_async` and
`CircuitBreaker.call` (which only react to exceptions) see it as a failure.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState, get_circuit_breaker
from .retry import RetryPolicy, retry_async
from config import settings

logger = logging.getLogger(__name__)


class ProviderCallError(Exception):
    """Raised when a provider call fails, whether by exception or by returning success=False."""


def _default_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=settings.provider_retry_max_attempts,
        base_delay=settings.provider_retry_base_delay_seconds,
        max_delay=settings.provider_retry_max_delay_seconds,
        # CircuitOpenError is deliberately excluded: retrying against an
        # already-open circuit just re-raises immediately every time, wasting
        # the retry budget and adding delay for no benefit - fail fast instead.
        retryable_exceptions=(ProviderCallError, TimeoutError, ConnectionError, OSError),
    )


class ResilientProvider:
    """
    Wraps a provider instance so its `generate_response` calls go through
    retry-with-backoff and a per-provider-type circuit breaker. All other
    methods (`initialize`, `close`) pass straight through to the wrapped
    instance; `health_check` additionally reports circuit breaker state
    (Task 8.4).
    """

    def __init__(self, inner: Any, retry_policy: Optional[RetryPolicy] = None, circuit_breaker: Optional[CircuitBreaker] = None):
        self._inner = inner
        self.name = getattr(inner, "name", inner.__class__.__name__)
        self.retry_policy = retry_policy or _default_retry_policy()
        self.circuit_breaker = circuit_breaker or get_circuit_breaker(self.name)

    async def initialize(self) -> bool:
        return await self._inner.initialize()

    async def close(self) -> None:
        return await self._inner.close()

    async def health_check(self) -> Dict[str, Any]:
        health = await self._inner.health_check()
        if isinstance(health, dict):
            health = dict(health)
            health["circuit_breaker"] = self.circuit_breaker.get_status()
        return health

    async def _call_and_raise_on_failure(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        response = await self._inner.generate_response(messages=messages, **kwargs)
        if not response.get("success", True):
            error = response.get("metadata", {}).get("error") or "provider call failed"
            raise ProviderCallError(error)
        return response

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        async def attempt() -> Dict[str, Any]:
            return await self.circuit_breaker.call(self._call_and_raise_on_failure, messages, **kwargs)

        try:
            return await retry_async(attempt, policy=self.retry_policy)
        except CircuitOpenError as e:
            logger.warning(f"Provider '{self.name}' call rejected: {e}")
            return {"content": "", "success": False, "metadata": {"error": str(e), "circuit_breaker": "open"}}
        except ProviderCallError as e:
            return {"content": "", "success": False, "metadata": {"error": str(e)}}

    async def stream_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Streaming counterpart to `generate_response` (Task 10.1/10.3).

        Deliberately simpler than the non-streaming path: no retry, since
        retrying a call that has already streamed some tokens to a client
        would mean either duplicating or discarding output they've already
        seen - neither is a sound recovery strategy. It still respects the
        circuit breaker: fails fast (without calling the inner provider at
        all) if the circuit is already open, and records a failure on the
        shared breaker if the stream raises before completing.

        Raises `CircuitOpenError` (if the circuit is open) or whatever the
        inner provider's stream raises - the caller
        (`WorkflowEngine.stream_execution`) is responsible for turning that
        into a client-visible error, since by the time a stream fails,
        tokens may already have been sent.
        """
        if not hasattr(self._inner, "stream_response"):
            raise NotImplementedError(f"Provider '{self.name}' does not support streaming")

        current_state = self.circuit_breaker.state
        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is open; refusing to start a new stream"
            )

        try:
            async for chunk in self._inner.stream_response(messages=messages, **kwargs):
                yield chunk
        except Exception:
            await self.circuit_breaker.record_failure()
            raise
        else:
            await self.circuit_breaker.record_success(current_state)
