"""
Tests for background/task_queue.py (Epic 5 / Story 5.1, Task 9.5).

Run with: pytest tests/test_background_queue.py -v
"""

import asyncio

import pytest

from background.task_queue import BackgroundTaskQueue


def test_enqueued_task_actually_runs():
    async def run():
        q = BackgroundTaskQueue(max_size=10, num_workers=1)
        await q.start()
        try:
            ran = asyncio.Event()

            async def job():
                ran.set()

            assert await q.enqueue(job, name="test-job") is True
            await asyncio.wait_for(ran.wait(), timeout=1.0)
            await q.stop(timeout=2.0)
            assert q.stats.completed == 1
        finally:
            if q._running:
                await q.stop(timeout=1.0)

    asyncio.run(run())


def test_caller_does_not_wait_for_background_work_to_complete():
    """
    Core Task 9.5 guarantee: enqueueing a slow job returns immediately - the
    caller (a request handler, in production) never blocks on it.
    """
    async def run():
        q = BackgroundTaskQueue(max_size=10, num_workers=1)
        await q.start()
        try:
            started = asyncio.Event()
            finished = asyncio.Event()

            async def slow_job():
                started.set()
                await asyncio.sleep(0.3)
                finished.set()

            enqueue_start = asyncio.get_event_loop().time()
            await q.enqueue(slow_job, name="slow-job")
            enqueue_duration = asyncio.get_event_loop().time() - enqueue_start

            assert enqueue_duration < 0.1, "enqueue() should return almost immediately"
            assert not finished.is_set(), "the slow job should not have finished yet"

            await asyncio.wait_for(finished.wait(), timeout=1.0)
        finally:
            await q.stop(timeout=2.0)

    asyncio.run(run())


def test_failing_background_task_does_not_crash_worker_or_app():
    """
    Core Task 9.5 guarantee: a background job raising doesn't take down its
    worker (which would silently stop draining the queue) or propagate
    anywhere the caller would see.
    """
    async def run():
        q = BackgroundTaskQueue(max_size=10, num_workers=1)
        await q.start()
        try:
            second_job_ran = asyncio.Event()

            async def failing_job():
                raise RuntimeError("boom")

            async def second_job():
                second_job_ran.set()

            await q.enqueue(failing_job, name="failing-job")
            await q.enqueue(second_job, name="second-job")

            # If the failure had crashed the worker, second_job would never run.
            await asyncio.wait_for(second_job_ran.wait(), timeout=1.0)
            await q.stop(timeout=2.0)
            assert q.stats.failed >= 1
            assert q.stats.completed >= 1
        finally:
            if q._running:
                await q.stop(timeout=1.0)

    asyncio.run(run())


def test_transient_failure_is_retried_then_succeeds():
    async def run():
        q = BackgroundTaskQueue(max_size=10, num_workers=1)
        await q.start()
        try:
            calls = {"count": 0}
            succeeded = asyncio.Event()

            async def flaky_job():
                calls["count"] += 1
                if calls["count"] < 2:
                    raise ConnectionError("transient")
                succeeded.set()

            await q.enqueue(flaky_job, name="flaky-job")
            await asyncio.wait_for(succeeded.wait(), timeout=2.0)
            await q.stop(timeout=2.0)

            assert calls["count"] == 2  # first attempt failed, retry succeeded
            assert q.stats.completed == 1
            assert q.stats.failed == 0  # eventual success isn't counted as a failure
        finally:
            if q._running:
                await q.stop(timeout=1.0)

    asyncio.run(run())


def test_enqueue_drops_and_reports_when_queue_is_full():
    async def run():
        q = BackgroundTaskQueue(max_size=1, num_workers=0)  # no workers -> nothing drains
        await q.start()
        try:
            async def noop():
                pass

            assert await q.enqueue(noop, name="job-1") is True
            assert await q.enqueue(noop, name="job-2") is False  # queue full, dropped
            assert q.stats.dropped_queue_full == 1
        finally:
            q._running = False  # no workers were started; avoid join() hanging on stop()

    asyncio.run(run())


def test_stop_is_idempotent_and_cancels_cleanly():
    async def run():
        q = BackgroundTaskQueue(max_size=10, num_workers=2)
        await q.start()
        await q.stop(timeout=2.0)
        # Calling stop() again (already stopped) must not raise.
        await q.stop(timeout=2.0)
        assert q._workers == []

    asyncio.run(run())


def test_get_status_shape():
    async def run():
        q = BackgroundTaskQueue(max_size=5, num_workers=2)
        status_before_start = q.get_status()
        assert status_before_start["running"] is False

        await q.start()
        status = q.get_status()
        assert status["running"] is True
        assert status["num_workers"] == 2
        assert status["max_size"] == 5
        await q.stop(timeout=2.0)

    asyncio.run(run())


def test_queue_can_be_restarted_across_separate_event_loops():
    """
    Regression test: BackgroundTaskQueue is used as a process-wide singleton
    (get_background_queue()), so it can be start()ed against a fresh event
    loop after a previous one has already closed (e.g. a new TestClient per
    test in this very suite). asyncio.Queue binds its internal
    locks/conditions to the loop running at construction time, so reusing
    one across loops raises confusing errors unless start() rebuilds it.
    """
    q = BackgroundTaskQueue(max_size=10, num_workers=1)

    async def run_once():
        await q.start()
        ran = asyncio.Event()

        async def job():
            ran.set()

        await q.enqueue(job, name="cross-loop-job")
        await asyncio.wait_for(ran.wait(), timeout=1.0)
        await q.stop(timeout=2.0)

    # Two separate asyncio.run() calls = two separate event loops, reusing
    # the same BackgroundTaskQueue instance - this is exactly the singleton
    # usage pattern in main.py across the whole test suite.
    asyncio.run(run_once())
    asyncio.run(run_once())
