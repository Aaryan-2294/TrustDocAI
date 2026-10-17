"""Verify documentation against the bundled sample project (temporary workspace; offline).

    python examples/verify_demo.py                 # a deliberately flawed generated document
    python examples/verify_demo.py --good          # an accurate one (expect no findings, score 100)
    python examples/verify_demo.py --file my.md    # your own Markdown against the sample code
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset
from trustdoc.ingestion import IngestionService
from trustdoc.models import Project
from trustdoc.verification import VerificationEngine, render_report

FLAWED = """The `compute_backoff` function accepts `attempt` and `base_delay`, and returns an integer.

Call `retry_all()` to requeue everything.

The signature is `def enqueue(payload, priority=1)`.

The `SqliteBackend` class persists tasks to disk. Failed tasks are retried up to three times by default.
"""
GOOD = """The `compute_backoff` function accepts `attempt`, `base_seconds` and `jitter`, and returns a float.

Use `enqueue(payload, priority=0, delay_seconds=0.0)` to add a task.

The `heartbeat` function returns a float.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", action="store_true")
    ap.add_argument("--file")
    args = ap.parse_args()
    text = pathlib.Path(args.file).read_text(encoding="utf-8") if args.file else (GOOD if args.good else FLAWED)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings.from_env({"TRUSTDOC_DATA_DIR": tmp, "TRUSTDOC_LOG_TO_FILE": "false", "TRUSTDOC_LOG_LEVEL": "WARNING"})
        settings.ensure_directories()
        db = Database(settings.storage.db_path)
        db.initialize()
        with db.session() as s:
            project_id = s.projects.create(Project(name="sample")).id
        IngestionService(db, settings).ingest_directory(project_id, load_dataset(DEFAULT_QUERIES).project_dir)
        report = VerificationEngine(db, settings).verify_text(project_id, text, name="generated/demo.md")
        print(render_report(report))


if __name__ == "__main__":
    main()
