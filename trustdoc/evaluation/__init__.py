"""Retrieval evaluation framework (Part 3).

Compares BM25-only, FAISS-only, hybrid (RRF) and hybrid + Cross-Encoder
pipelines with Recall@K and MRR over a labelled query set. Imports are lazy so a
core install can still import this package.
"""

from __future__ import annotations

import importlib

_LAZY = {
    "run_sample_evaluation": "runner", "evaluate": "runner", "write_report": "runner",
    "render_summary": "runner", "report_to_dict": "runner", "PIPELINES": "runner",
    "EvaluationReport": "runner", "load_dataset": "dataset", "EvalDataset": "dataset",
    "EvaluationError": "dataset", "recall_at_k": "metrics", "reciprocal_rank": "metrics",
}
__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(f"trustdoc.evaluation.{_LAZY[name]}"), name)
    raise AttributeError(f"module 'trustdoc.evaluation' has no attribute {name!r}")
