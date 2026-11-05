"""
Retry with exponential backoff for transient failures.

Distinct from WorkflowEngine's confidence-based validation retry loop (see
package docstring in `brain/resilience/__init__.py`) - this retries on
*transport* failures (network errors, timeouts, a provider reporting
`success: False`), not on low response quality.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """
    Exponential backoff parameters: delay for attempt *n* (1-indexed) is
    `min(base_delay * (exponential_base ** (n - 1)), max_delay)`, plus a
    small random jitter to avoid many concurrent callers retrying in lockstep.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: Tuple[Type[BaseException], ...] = field(default=(Exception,))

    def delay_for_attempt(self, attempt: int) -> float:
        """`attempt` is 1-indexed (the attempt that just failed)."""
        delay = min(self.base_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)
        return delay + random.uniform(0, self.jitter)


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Call `func(*args, **kwargs)`, retrying up to `policy.max_attempts` times
    (with exponential backoff) on any of `policy.retryable_exceptions`.
    Re-raises the last exception once attempts are exhausted.
    """
    policy = policy or RetryPolicy()
    last_exc: BaseException = RuntimeError("retry_async called with max_attempts <= 0")

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except policy.retryable_exceptions as e:
            last_exc = e
            if attempt >= policy.max_attempts:
                logger.warning(f"Retry exhausted after {attempt} attempt(s): {e}")
                break
            delay = policy.delay_for_attempt(attempt)
            if on_retry is not None:
                on_retry(attempt, e)
            logger.warning(
                f"Attempt {attempt}/{policy.max_attempts} failed ({e}); "
                f"retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)

    raise last_exc
