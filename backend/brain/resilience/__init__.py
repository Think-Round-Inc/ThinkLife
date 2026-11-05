"""
Structured error handling: retry with backoff + circuit breaker.

Epic 5 / Story 5.2, Task 8. This is deliberately separate from
`WorkflowEngine`'s existing confidence-based validation retry loop
(`brain/cortex/workflow_engine.py`, governed by `settings.
max_validation_attempts` / `settings.confidence_threshold`) - that loop
retries when a response's *quality* is too low; this package retries when a
provider call *transport-fails* (timeout, connection error, or the provider
adapter itself reporting `success: False`).

See `provider_resilience.py` for how the two pieces (retry + circuit
breaker) are combined and applied at the provider boundary.
"""

from .retry import RetryPolicy, retry_async
from .circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError, get_circuit_breaker, all_circuit_breaker_status
from .provider_resilience import ResilientProvider, ProviderCallError
from .tool_resilience import run_tool_with_retry, ToolCallError

__all__ = [
    "RetryPolicy",
    "retry_async",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "get_circuit_breaker",
    "all_circuit_breaker_status",
    "ResilientProvider",
    "ProviderCallError",
    "run_tool_with_retry",
    "ToolCallError",
]
