"""Ask a question about the bundled sample project (works in a temporary workspace).

    # offline smoke test: lexical stand-in retrieval + extractive stub "LLM" (NOT real models)
    python examples/ask_demo.py "compute_backoff"

    # real retrieval models + your configured local LLM (e.g. TRUSTDOC_LLM_PROVIDER=ollama):
    python examples/ask_demo.py --real-retrieval --llm configured "How are retries delayed?"

    # force an abstention to see the insufficient-evidence path (model is never called)
    python examples/ask_demo.py --threshold 1.5 "compute_backoff"
"""

from __future__ import annotations

import argparse
import tempfile

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset
from trustdoc.ingestion import IngestionService
from trustdoc.llm.stub import ExtractiveStubLLM
from trustdoc.models import Project
from trustdoc.retrieval import HashingEmbedder, OverlapReranker, create_retriever
from trustdoc.workflows import AskService, render_ask_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--real-retrieval", action="store_true", help="use the configured Sentence-Transformer + Cross-Encoder")
    ap.add_argument("--llm", choices=["stub", "configured"], default="stub")
    ap.add_argument("--threshold", type=float, default=None, help="override the relevance threshold")
    ap.add_argument("--show-evidence", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        env = {"TRUSTDOC_DATA_DIR": tmp, "TRUSTDOC_LOG_TO_FILE": "false", "TRUSTDOC_LOG_LEVEL": "WARNING"}
        if args.threshold is not None:
            env["TRUSTDOC_RELEVANCE_THRESHOLD"] = str(args.threshold)
        import os
        for key in os.environ:                       # keep the user's LLM / limits settings when asked
            if args.llm == "configured" and key.startswith(("TRUSTDOC_LLM_", "TRUSTDOC_ASK_")):
                env[key] = os.environ[key]
        settings = Settings.from_env(env)
        settings.ensure_directories()
        db = Database(settings.storage.db_path)
        db.initialize()
        with db.session() as s:
            project_id = s.projects.create(Project(name="sample")).id
        batch = IngestionService(db, settings).ingest_directory(project_id, load_dataset(DEFAULT_QUERIES).project_dir)
        print(f"Ingested {len(batch.results)} sources into a temporary project.\n")

        if args.real_retrieval:
            retriever = create_retriever(db, settings)
        else:
            retriever = create_retriever(db, settings, embedder=HashingEmbedder(), reranker=OverlapReranker())
        llm = ExtractiveStubLLM() if args.llm == "stub" else None       # None -> built from TRUSTDOC_LLM_* settings
        result = AskService(db, settings, retriever, llm).ask(project_id, args.question)
        print(render_ask_result(result, show_evidence=args.show_evidence))


if __name__ == "__main__":
    main()
