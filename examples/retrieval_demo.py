"""Index the bundled sample project and compare the four retrieval modes on one query.

    python examples/retrieval_demo.py "retry delay between attempts"
    python examples/retrieval_demo.py --embedder hashing --reranker overlap "compute_backoff"

Default uses the real configured models (needs `pip install -e ".[retrieval,pdf]"` and a
one-time model download). `--embedder hashing --reranker overlap` runs offline with lexical
STAND-INS, which demonstrates the plumbing but says nothing about real-model quality.
Works in a temporary workspace; your own data directory is not touched.
"""

from __future__ import annotations

import argparse
import json
import tempfile

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset
from trustdoc.ingestion import IngestionService
from trustdoc.models import Project
from trustdoc.retrieval import (
    HashingEmbedder,
    IndexManager,
    OverlapReranker,
    RetrievalMode,
    Retriever,
    create_embedder,
    create_reranker,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--embedder", choices=["sentence-transformers", "hashing"], default="sentence-transformers")
    ap.add_argument("--reranker", choices=["cross-encoder", "overlap"], default="cross-encoder")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings.from_env({"TRUSTDOC_DATA_DIR": tmp, "TRUSTDOC_LOG_TO_FILE": "false",
                                      "TRUSTDOC_LOG_LEVEL": "WARNING"})
        settings.ensure_directories()
        db = Database(settings.storage.db_path)
        db.initialize()
        with db.session() as s:
            pid = s.projects.create(Project(name="demo")).id
        IngestionService(db, settings).ingest_directory(pid, load_dataset(DEFAULT_QUERIES).project_dir)

        embedder = HashingEmbedder() if args.embedder == "hashing" else create_embedder(settings)
        reranker = OverlapReranker() if args.reranker == "overlap" else create_reranker(settings)
        manager = IndexManager(db, settings, embedder)
        retriever = Retriever(db, settings, manager, embedder, reranker)

        m = manager.build(pid)                       # indexing is an explicit, one-time operation
        print(f"Indexed project: {m.chunk_count} chunks from {m.source_count} sources | "
              f"model={m.embedding_model} dim={m.embedding_dimension} | {m.faiss_metric}")
        print(f"Index files: {sorted(p.name for p in manager.index_dir(pid).iterdir())}\n")

        for mode in RetrievalMode:
            res = retriever.search(pid, args.query, top_k=args.top_k, mode=mode, apply_threshold=False)
            print(f"=== {mode.value}  (query: {args.query!r}, {res.timings_ms['total_ms']:.1f} ms)")
            for e in res.ranked:
                parts = [f"#{e.final_rank} {e.citation.label}"]
                if e.bm25_rank: parts.append(f"bm25 r{e.bm25_rank} s={e.bm25_score:.2f}")
                if e.faiss_rank: parts.append(f"faiss r{e.faiss_rank} cos={e.faiss_score:.3f}")
                if e.rrf_score is not None: parts.append(f"rrf r{e.rrf_rank} s={e.rrf_score:.4f}")
                if e.rerank_score is not None: parts.append(f"rerank={e.rerank_score:.3f}")
                print("   " + " | ".join(parts))
            print()

        res = retriever.search(pid, args.query, top_k=1, apply_threshold=False)
        print("Full evidence record for the top result (provenance intact):")
        print(json.dumps(res.ranked[0].to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
