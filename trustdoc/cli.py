"""Command-line entry point: ``python -m trustdoc <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
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


def _ask(args: argparse.Namespace) -> int:
    """Answer a question about an ingested project.

    Exit codes: 0 answered or "insufficient evidence"; 1 operational failure (retrieval or
    generation); 2 usage/configuration problem (unknown project, invalid question, no model
    configured, missing dependency).
    """
    from trustdoc.exceptions import NotFoundError
    from trustdoc.workflows import AskStatus, create_ask_service, render_ask_result

    settings = Settings.from_env()
    if args.provider or args.model:
        llm_cfg = dataclasses.replace(
            settings.llm,
            provider=args.provider or settings.llm.provider,
            model=args.model if args.model is not None else settings.llm.model,
        )
        settings = dataclasses.replace(settings, llm=llm_cfg)
    ctx = bootstrap(settings)

    project_id = args.project_id
    if project_id is None:
        with ctx.db.session() as s:
            project = s.projects.get_by_name(args.project)
        if project is None:
            raise NotFoundError(f"No project named {args.project!r}")
        project_id = project.id

    service = create_ask_service(ctx.db, ctx.settings)
    result = service.ask(project_id, args.question, top_k=args.top_k)

    if args.json:
        print(json.dumps(result.to_dict(include_evidence_text=False), indent=2))
    else:
        text = render_ask_result(result, show_evidence=args.show_evidence)
        print(text, file=sys.stderr if result.status in (AskStatus.RETRIEVAL_ERROR, AskStatus.GENERATION_ERROR) else sys.stdout)

    if result.status in (AskStatus.RETRIEVAL_ERROR, AskStatus.GENERATION_ERROR):
        return 2 if result.error_kind in ("not_configured", "dependency_missing") else 1
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
    ask = sub.add_parser(
        "ask",
        help="answer a question about an ingested project, with citations (needs a local LLM)",
        description="Retrieve evidence from a project and answer using the configured local LLM. "
                    "Configure the model with TRUSTDOC_LLM_PROVIDER / TRUSTDOC_LLM_MODEL "
                    "(see .env.example). If retrieval finds insufficient evidence the model is not called.",
    )
    target = ask.add_mutually_exclusive_group(required=True)
    target.add_argument("--project-id", type=int, help="id of the project to query")
    target.add_argument("--project", help="name of the project to query")
    ask.add_argument("--question", "-q", required=True, help="the question to answer")
    ask.add_argument("--top-k", type=int, default=None, help="evidence items to retrieve (default: config)")
    ask.add_argument("--provider", default=None, help="override TRUSTDOC_LLM_PROVIDER (ollama|transformers|stub)")
    ask.add_argument("--model", default=None, help="override TRUSTDOC_LLM_MODEL")
    ask.add_argument("--show-evidence", action="store_true", help="also print short snippets of the evidence used")
    ask.add_argument("--json", action="store_true", help="print the full structured result as JSON")
    args = parser.parse_args(argv)

    try:
        if args.command == "eval-retrieval":
            return _eval_retrieval(args)
        if args.command == "ask":
            return _ask(args)
        ctx = bootstrap(Settings.from_env())
    except TrustDocError as exc:
        print(f"error: {exc}", file=sys.stderr if args.command == "ask" else sys.stdout)
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
