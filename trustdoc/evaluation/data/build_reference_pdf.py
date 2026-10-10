"""Regenerate sample_project/reference.pdf (requires PyMuPDF). Page texts are the fixture."""
import pathlib

import pymupdf

PAGES = [
    "TaskQueue Technical Reference\n\n1. Delivery guarantees\n\n"
    "TaskQueue provides at-least-once delivery. A task is handed to a worker, and the worker must "
    "acknowledge it before the visibility timeout expires. If a worker crashes before acknowledging, "
    "the task becomes visible again and another worker receives it, so a task can be delivered more "
    "than once. Handlers must therefore be idempotent.",
    "2. Retry and backoff\n\n"
    "Retries use exponential backoff. The delay before attempt n is base * 2^n seconds, capped at "
    "300 seconds. Full jitter then picks a random delay between zero and that value, which spreads "
    "retries out and prevents many failed tasks from hitting a dependency at the same moment (the "
    "thundering herd problem). After the maximum number of attempts a task stays in the failed set.",
    "3. Persistence and compaction\n\n"
    "The SQLite backend runs in write-ahead logging mode. Completed tasks are retained for seven "
    "days and are then deleted by compaction. Deleting rows does not shrink the database file; a "
    "VACUUM rewrites the file and returns the unused space to the operating system.",
]

doc = pymupdf.open()
for text in PAGES:
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
doc.set_metadata({"title": "TaskQueue Technical Reference", "author": "TrustDoc evaluation fixture"})
out = pathlib.Path(__file__).parent / "sample_project" / "reference.pdf"
out.write_bytes(doc.tobytes(deflate=True, no_new_id=True))
print("wrote", out)
