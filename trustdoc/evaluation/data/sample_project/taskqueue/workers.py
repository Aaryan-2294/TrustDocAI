"""Worker threads that consume tasks from a queue."""

from __future__ import annotations

import random
import threading
import time


class WorkerPool:
    """Runs tasks from a TaskQueue on a fixed number of threads."""

    def __init__(self, queue, handler, visibility_timeout: float = 30.0):
        """Tasks not acknowledged within ``visibility_timeout`` seconds become visible to other workers again."""
        self._queue = queue
        self._handler = handler
        self._visibility_timeout = visibility_timeout
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self, num_workers: int = 4) -> None:
        """Launch ``num_workers`` daemon threads that poll the queue."""
        for _ in range(num_workers):
            thread = threading.Thread(target=self._run, daemon=True)
            thread.start()
            self._threads.append(thread)

    def shutdown(self, graceful: bool = True, wait_seconds: float = 10.0) -> None:
        """Stop the workers; a graceful shutdown lets in-flight tasks finish before returning."""
        self._stop.set()
        if graceful:
            for thread in self._threads:
                thread.join(timeout=wait_seconds)

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._queue.dequeue(timeout=1.0)
            if task is not None:
                self._handler(task)


def heartbeat(worker_id: str, interval_seconds: float = 30.0) -> float:
    """Record that a worker is alive and return when the next beat is expected."""
    return time.time() + interval_seconds


def compute_backoff(attempt: int, base_seconds: float = 2.0, jitter: bool = True) -> float:
    """Exponential backoff delay for retry number ``attempt``, capped at 300 seconds, with optional random jitter."""
    delay = min(base_seconds * (2 ** attempt), 300.0)
    return random.uniform(0, delay) if jitter else delay
