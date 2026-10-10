"""Priority task queue with delayed delivery and bounded retries."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

DEFAULT_MAX_ATTEMPTS = 3


@dataclass(order=True)
class Task:
    """A unit of work waiting in the queue."""

    priority: int
    available_at: float
    payload: dict = field(compare=False, default_factory=dict)
    attempts: int = field(compare=False, default=0)


class TaskQueue:
    """In-memory priority queue; lower priority numbers are served first."""

    def __init__(self, backend=None, max_size: int = 10_000):
        """Create a queue, optionally persisting tasks through ``backend``."""
        self._heap: list[Task] = []
        self._failed: list[Task] = []
        self._backend = backend
        self._max_size = max_size

    def enqueue(self, payload: dict, priority: int = 0, delay_seconds: float = 0.0) -> Task:
        """Add a task. A positive ``delay_seconds`` hides it until the delay has elapsed."""
        if len(self._heap) >= self._max_size:
            raise OverflowError("queue is full")
        task = Task(priority, time.time() + delay_seconds, payload)
        heapq.heappush(self._heap, task)
        return task

    def dequeue(self, timeout: float | None = None) -> Task | None:
        """Return the next available task, waiting up to ``timeout`` seconds for one."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            now = time.time()
            if self._heap and self._heap[0].available_at <= now:
                return heapq.heappop(self._heap)
            if deadline is not None and now >= deadline:
                return None
            time.sleep(0.05)

    def retry_failed(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> int:
        """Requeue failed tasks that still have attempts left; return how many were requeued."""
        requeued = 0
        for task in list(self._failed):
            if task.attempts < max_attempts:
                self._failed.remove(task)
                heapq.heappush(self._heap, task)
                requeued += 1
        return requeued

    def purge(self) -> int:
        """Drop every pending task and return the number of tasks removed."""
        removed = len(self._heap)
        self._heap.clear()
        return removed
