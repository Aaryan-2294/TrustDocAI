"""Command-line entry point: ``python -m trustdoc <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trustdoc import __version__
from trustdoc.app import bootstrap
from trustdoc.config import Settings
from trustdoc.exceptions import TrustDocError


def _eval_retrieval(args: argparse.Namespace) -> int:
    """Run the retrieval evaluation in an isolated temp workspace (your DB is untouched)."""
    from trustdoc.evaluation import run_sample_evaluation
    from trustdoc.evaluation.dataset import DEFAULT_QUERIES
    from trustdoc.evaluation.runner import render_summary
    from trustdoc.retrieval.embedding import HashingEmbedder
    from trustdoc.retrieval.factory import create_embedder, create_reranker
    from trustdoc.retrieval.reranker import OverlapReranker

    settings = Settings.from_env()
    embedder = HashingEmbedder() if args.embedder == "hashing" else create_embedder(settings)
    reranker = {"none": None, "overlap": OverlapReranker()}.get(args.reranker)
    if args.reranker == "cross-encoder":
        reranker = create_reranker(settings)
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    report = run_sample_evaluation(
        args.out, embedder, reranker, queries_path=args.queries or DEFAULT_QUERIES, ks=ks
    )
    print(render_summary(report))
    print(f"Wrote results.json, per_query.csv and summary.md to {Path(args.out).resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trustdoc", description="TrustDoc AI command-line tools")
    parser.add_argument("--version", action="version", version=f"trustdoc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create data directories and the SQLite database")
    sub.add_parser("info", help="show resolved settings, schema version and row counts")
    ev = sub.add_parser(
        "eval-retrieval",
        help="compare BM25 / FAISS / hybrid / hybrid+rerank on the bundled sample project",
    )
    ev.add_argument("--embedder", choices=["sentence-transformers", "hashing"],
                    default="sentence-transformers",
                    help="'hashing' is a lexical stand-in (offline smoke runs, NOT real-model numbers)")
    ev.add_argument("--reranker", choices=["cross-encoder", "overlap", "none"], default="cross-encoder")
    ev.add_argument("--out", default="evaluation_results/retrieval", help="output directory")
    ev.add_argument("--ks", default="1,3,5,10", help="comma-separated K values for Recall@K")
    ev.add_argument("--queries", default=None, help="path to a custom queries.json")
    args = parser.parse_args(argv)

    try:
        if args.command == "eval-retrieval":
            return _eval_retrieval(args)
        ctx = bootstrap(Settings.from_env())
    except TrustDocError as exc:
        print(f"error: {exc}")
        return 2

    if args.command == "init":
        print(f"Initialised database at {ctx.settings.storage.db_path} "
              f"(schema v{ctx.db.schema_version()})")
    elif args.command == "info":
        print(json.dumps(
            {
                "version": __version__,
                "schema_version": ctx.db.schema_version(),
                "settings": ctx.settings.to_dict(),
                "table_counts": ctx.db.table_counts(),
            },
            indent=2,
        ))
    return 0
