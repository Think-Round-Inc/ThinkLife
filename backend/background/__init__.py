"""
Background task execution layer (Epic 5 / Story 5.1, Task 9).

    from background import get_background_queue
"""

from .task_queue import BackgroundTaskQueue, BackgroundQueueStats, get_background_queue

__all__ = ["BackgroundTaskQueue", "BackgroundQueueStats", "get_background_queue"]
