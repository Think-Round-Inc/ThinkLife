"""
Background task execution layer.

Epic 5 / Story 5.1, Task 9. Chosen mechanism (9.1): a bounded `asyncio.Queue`
with a small pool of worker tasks, rather than bare FastAPI `BackgroundTasks`.
`BackgroundTasks` has no backpressure control - under a burst of concurrent
requests it would spawn an unbounded number of concurrent background
coroutines, which is exactly the kind of resource-exhaustion risk Story 5.1
is about *preventing*, not introducing. A bounded queue with a fixed worker
count means background load is capped and predictable regardless of request
volume; `enqueue()` drops (with a warning, not an exception) rather than
blocking the caller when the queue is full, since this is for best-effort
work the request doesn't need to wait on.

Usage (see main.py's `lifespan` for start/stop, Task 9.3):

    from background import get_background_queue

    queue = get_background_queue()
    await queue.start()
    ...
    await queue.enqueue(some_async_func, arg1, arg2, name="evaluate_response")
    ...
    await queue.stop()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from brain.resilience import RetryPolicy, retry_async
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class _QueueItem:
    func: Callable[..., Awaitable[Any]]
    args: tuple
    kwargs: dict
    name: str = "background_task"


@dataclass
class BackgroundQueueStats:
    enqueued: int = 0
    dropped_queue_full: int = 0
    completed: int = 0
    failed: int = 0


class BackgroundTaskQueue:
    """
    A bounded queue of async callables, drained by a small pool of worker
    tasks. Each job gets its own retry budget (Task 9.4) and a failure never
    crashes a worker or the app (Task 9.5) - it's logged and counted instead.
    """

    def __init__(self, max_size: Optional[int] = None, num_workers: Optional[int] = None):
        self.max_size = max_size or settings.background_queue_max_size
        self.num_workers = num_workers or settings.background_queue_num_workers
        self._queue: "asyncio.Queue[Optional[_QueueItem]]" = asyncio.Queue(maxsize=self.max_size)
        self._workers: list = []
        self._running = False
        self.stats = BackgroundQueueStats()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Recreate the queue fresh on every start(), bound to whichever
        # event loop is running now. asyncio.Queue's internal
        # locks/conditions bind to the loop active at construction time; as
        # a process-wide singleton (get_background_queue()), this object can
        # outlive the event loop it was first created on (e.g. a fresh
        # TestClient per test, or an app restart within the same process),
        # and reusing a Queue built on a now-closed loop raises confusing
        # cross-loop errors. Any items from a previous run are, by design,
        # already gone once stop() has drained/cancelled everything.
        self._queue = asyncio.Queue(maxsize=self.max_size)
        self._workers = [
            asyncio.create_task(self._worker_loop(i), name=f"bg-worker-{i}")
            for i in range(self.num_workers)
        ]
        logger.info(f"Background task queue started with {self.num_workers} worker(s)")

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop accepting new work implicitly (callers should stop calling
        `enqueue` themselves), drain what's already queued up to `timeout`
        seconds, then cancel any workers still running and await their
        cancellation (Task 9.3 - clean shutdown, no dangling tasks).
        """
        if not self._running:
            return
        self._running = False

        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"Background queue did not drain within {timeout}s "
                f"({self._queue.qsize()} item(s) still queued) - cancelling workers"
            )

        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        logger.info(
            f"Background task queue stopped (completed={self.stats.completed}, "
            f"failed={self.stats.failed}, dropped={self.stats.dropped_queue_full})"
        )

    async def enqueue(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        name: str = "background_task",
        **kwargs: Any,
    ) -> bool:
        """
        Queue `func(*args, **kwargs)` to run on a worker. Returns True if
        queued, False if the queue was full (dropped, not blocked - see
        module docstring).
        """
        item = _QueueItem(func=func, args=args, kwargs=kwargs, name=name)
        try:
            self._queue.put_nowait(item)
            self.stats.enqueued += 1
            return True
        except asyncio.QueueFull:
            self.stats.dropped_queue_full += 1
            logger.warning(f"Background queue full (max_size={self.max_size}); dropping task '{name}'")
            return False

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:  # sentinel, not currently used but supported
                    break
                await self._run_item(item)
            finally:
                self._queue.task_done()

    async def _run_item(self, item: _QueueItem) -> None:
        policy = RetryPolicy(
            max_attempts=settings.background_task_retry_max_attempts,
            base_delay=settings.background_task_retry_base_delay_seconds,
            max_delay=settings.background_task_retry_base_delay_seconds * 4,
            retryable_exceptions=(Exception,),
        )
        try:
            await retry_async(item.func, *item.args, policy=policy, **item.kwargs)
            self.stats.completed += 1
        except Exception:
            # A background job failing must never crash the worker (which
            # would silently stop draining the queue) or the app (Task 9.5).
            self.stats.failed += 1
            logger.exception(f"Background task '{item.name}' failed after retries")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "num_workers": self.num_workers,
            "queue_size": self._queue.qsize(),
            "max_size": self.max_size,
            "enqueued": self.stats.enqueued,
            "completed": self.stats.completed,
            "failed": self.stats.failed,
            "dropped_queue_full": self.stats.dropped_queue_full,
        }


_background_queue_instance: Optional[BackgroundTaskQueue] = None


def get_background_queue() -> BackgroundTaskQueue:
    """Process-wide singleton, mirroring the pattern used for CortexFlow/WorkflowEngine."""
    global _background_queue_instance
    if _background_queue_instance is None:
        _background_queue_instance = BackgroundTaskQueue()
    return _background_queue_instance
