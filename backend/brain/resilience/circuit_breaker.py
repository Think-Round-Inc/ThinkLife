"""
Circuit breaker for provider calls.

Standard three-state design (CLOSED -> OPEN -> HALF_OPEN -> CLOSED):
- CLOSED: calls pass through normally; consecutive failures are counted.
- OPEN: calls are rejected immediately (`CircuitOpenError`) without touching
  the provider, once `failure_threshold` consecutive failures occur.
- HALF_OPEN: after `recovery_timeout` seconds, the next call(s) are let
  through as a trial. Success closes the circuit again; failure re-opens it.

Known simplification: under concurrent load, more than one call may observe
HALF_OPEN and be let through as a "trial" simultaneously (this isn't a
single-token gate). That's an acceptable trade-off for a lightweight,
in-process breaker rather than a strict single-trial implementation - a
handful of concurrent trial calls against a recovering provider is a minor
cost, not a correctness issue.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.recovery_timeout = recovery_timeout or settings.circuit_breaker_recovery_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for recovery-timeout-based OPEN -> HALF_OPEN transition."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.time() - self._opened_at >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Run `func(*args, **kwargs)` through the breaker. Raises
        `CircuitOpenError` immediately (without calling `func`) if the
        circuit is open and the recovery timeout hasn't elapsed yet.
        """
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is open (failed {self._failure_count} times); "
                f"retry after recovery timeout"
            )

        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self.record_failure()
            raise
        else:
            await self.record_success(current_state)
            return result

    async def record_failure(self) -> None:
        """
        Public entry point for recording a failure without going through
        `call()` - used by callers wrapping an async generator (streaming)
        rather than a single awaitable, where `call()`'s shape doesn't fit.
        """
        await self._record_failure()

    async def record_success(self, observed_state: CircuitState) -> None:
        """Counterpart to `record_failure()` - see its docstring."""
        await self._record_success(observed_state)

    async def _record_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.warning(
                        f"Circuit '{self.name}' opening after {self._failure_count} consecutive failures"
                    )
                self._state = CircuitState.OPEN
                self._opened_at = time.time()

    async def _record_success(self, observed_state: CircuitState) -> None:
        async with self._lock:
            if observed_state == CircuitState.HALF_OPEN or self._state == CircuitState.OPEN:
                logger.info(f"Circuit '{self.name}' closing after a successful trial call")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._opened_at = None
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout,
        }


_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = asyncio.Lock()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Get (creating if needed) the process-wide circuit breaker for a given
    name (typically a provider type like "openai"). Not itself async since
    dict get-or-create on a plain dict is safe without an await in between;
    the lock exists for callers that want to coordinate around it.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name)
    return _circuit_breakers[name]


def all_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Status of every circuit breaker created so far - used by health checks (Task 8.4)."""
    return {name: cb.get_status() for name, cb in _circuit_breakers.items()}
