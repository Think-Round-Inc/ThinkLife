"""
Retry wrapper for tool execution.

Per the Task 8 plan: the retry policy is meant for "transient provider/tool
failures" (both), while the circuit breaker is explicitly "per provider"
only. So tools get retry-with-backoff (this module) but not a circuit
breaker - a single flaky tool call doesn't warrant short-circuiting future
calls to that tool the way a struggling LLM provider does.

Same adaptation as `provider_resilience.py`: tool `execute()` implementations
(see `brain/tools/tavily_search.py`) catch their own exceptions and return a
`ToolResult(success=False, ...)` rather than raising, so `ToolCallError`
bridges that into something `retry_async` can react to.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from .retry import RetryPolicy, retry_async
from config import settings

if TYPE_CHECKING:
    from ..tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolCallError(Exception):
    """Raised when a tool call fails, whether by exception or by returning success=False."""


def _default_tool_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=settings.tool_retry_max_attempts,
        base_delay=settings.tool_retry_base_delay_seconds,
        max_delay=settings.tool_retry_max_delay_seconds,
        retryable_exceptions=(ToolCallError, TimeoutError, ConnectionError, OSError),
    )


async def run_tool_with_retry(
    tool: "BaseTool",
    timeout_seconds: float,
    retry_policy: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> "ToolResult":
    """
    Run a tool through `run_tool_with_timeout` (brain/lifecycle.py, Task 6 -
    which still guarantees initialize()/close() pairing on every attempt),
    retrying on transient failures per `retry_policy` (default: Task 8's
    tool retry settings).

    A `ToolResult(success=False, ...)` from the tool is converted into a
    `ToolCallError` so it's retried like any other transient failure; if
    retries are exhausted, the last failing `ToolResult` is returned to the
    caller rather than raising, matching the pre-existing behavior where
    `_execute_one_tool` (workflow_engine.py) always gets back a `ToolResult`
    (or a real timeout/exception) to build its response dict from.
    """
    from .. import lifecycle as lifecycle_module
    from ..tools.base_tool import ToolResult

    policy = retry_policy or _default_tool_retry_policy()
    last_failed_result: Optional["ToolResult"] = None

    async def attempt() -> "ToolResult":
        nonlocal last_failed_result
        result = await lifecycle_module.run_tool_with_timeout(tool, timeout_seconds, **kwargs)
        if not result.success:
            last_failed_result = result
            raise ToolCallError(result.error or "tool execution failed")
        return result

    try:
        return await retry_async(attempt, policy=policy)
    except ToolCallError:
        # Retries exhausted on a returned ToolResult(success=False, ...);
        # return that result rather than propagating the bridge exception -
        # callers only know about ToolResult, not this module's internals.
        assert last_failed_result is not None
        return last_failed_result
    except (TimeoutError, ConnectionError, OSError) as e:
        # Retries exhausted on a *raised* exception (e.g. run_tool_with_timeout's
        # asyncio.wait_for timing out) rather than a returned failure object -
        # there's no ToolResult to hand back here, so build one. TimeoutError
        # has no message by default, so fall back to its type name.
        tool_name = getattr(tool, "name", tool.__class__.__name__)
        error_message = str(e) or f"{type(e).__name__} after retries exhausted"
        return ToolResult(tool_name=tool_name, success=False, content=None, metadata={}, error=error_message)
