# TaskQueue

A small embeddable priority task queue for Python services.

## Installation

Install the library with `pip install taskqueue-lite`. It requires Python 3.9 or newer and has no
third-party dependencies.

## Quick start

Create a `TaskQueue`, enqueue payloads, and let a `WorkerPool` run your handler on background threads.

## Configuration

Settings are read from a JSON file by `load_config`. Environment variables prefixed with `TASKQ_`
override the file: `TASKQ_WORKER_COUNT` sets the number of workers, `TASKQ_VISIBILITY_TIMEOUT` sets
how long an unacknowledged task stays hidden, and `TASKQ_DB_PATH` selects the SQLite file.

## Retry policy

Failed tasks are retried up to three times by default. Call `retry_failed` to requeue them; the
delay between attempts grows exponentially so a struggling dependency is not hammered.

## Troubleshooting

### Tasks never run

Check that the pool was started and that `worker_count` is at least 1. Delayed tasks stay hidden
until their delay has elapsed.

### Database file keeps growing

Completed tasks are only removed when compaction runs. Call `compact` regularly, and pass
`vacuum=True` occasionally to shrink the file on disk.
