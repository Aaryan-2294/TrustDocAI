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


def _verify(args: argparse.Namespace) -> int:
    """Verify documentation against the project's code.

    Exit codes: 0 completed, no findings at/above --fail-on; 3 completed, documentation findings at/above
    --fail-on (a documentation problem, NOT a crash); 2 usage/configuration problem (unknown project or
    document, ...); 1 operational failure (the verification run itself failed).
    """
    from trustdoc.exceptions import NotFoundError, VerificationError
    from trustdoc.ingestion import IngestionService
    from trustdoc.models import Severity
    from trustdoc.verification import VerificationEngine, render_report

    settings = Settings.from_env()
    if args.provider or args.model:
        llm_cfg = dataclasses.replace(
            settings.llm, provider=args.provider or settings.llm.provider,
            model=args.model if args.model is not None else settings.llm.model)
        settings = dataclasses.replace(settings, llm=llm_cfg)
    ctx = bootstrap(settings)
    engine = VerificationEngine(ctx.db, ctx.settings)

    try:
        if args.show_run is not None:
            report = engine.load_report(args.show_run)
            if args.project_id is not None and report.project_id != args.project_id:
                raise NotFoundError(f"Run {args.show_run} does not belong to project {args.project_id}")
        else:
            project_id = args.project_id
            if project_id is None:
                with ctx.db.session() as s:
                    project = s.projects.get_by_name(args.project)
                if project is None:
                    raise NotFoundError(f"No project named {args.project!r}")
                project_id = project.id
            documents = list(args.document or [])
            for path in args.document_file or []:
                documents.append(IngestionService(ctx.db, ctx.settings).ingest_file(project_id, path).source.name)
            report = engine.verify_project(project_id, documents=documents or None,
                                           include_docstrings=not args.no_docstrings, semantic=args.semantic)
    except VerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report, max_per_severity=args.max_findings))
    if args.fail_on != "none":
        worst = report.max_severity()
        if worst is not None and worst.rank >= Severity(args.fail_on).rank:
            return 3
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
    ver = sub.add_parser(
        "verify",
        help="check documentation against the code: contradictions, unverified claims, missing docs, Trust Score",
        description="Extract claims from Markdown documentation and Python docstrings and verify them. Tier 1 (always on) is "
                    "deterministic static analysis of the indexed code. Tier 2 (--semantic) is optional, LLM-assisted and uses "
                    "retrieved evidence; it needs a configured local LLM and is never ground truth. Missing evidence is reported "
                    "as 'insufficient evidence', never as 'false'.",
    )
    vt = ver.add_mutually_exclusive_group()
    vt.add_argument("--project-id", type=int, help="id of the project to verify")
    vt.add_argument("--project", help="name of the project to verify")
    ver.add_argument("--document", action="append", metavar="NAME",
                     help="verify only this already-ingested Markdown source (project-relative name); repeatable")
    ver.add_argument("--document-file", action="append", metavar="PATH",
                     help="ingest this Markdown file into the project first, then verify it; repeatable")
    ver.add_argument("--no-docstrings", action="store_true", help="do not check Python docstrings")
    ver.add_argument("--semantic", action="store_true", help="also run LLM-assisted semantic checks on behaviour/config claims")
    ver.add_argument("--provider", default=None, help="override TRUSTDOC_LLM_PROVIDER (semantic checks only)")
    ver.add_argument("--model", default=None, help="override TRUSTDOC_LLM_MODEL (semantic checks only)")
    ver.add_argument("--fail-on", choices=["none", "critical", "high", "medium", "low"], default="high",
                     help="exit with status 3 if any finding at or above this severity exists (default: high; INFO never fails)")
    ver.add_argument("--max-findings", type=int, default=10, help="findings shown per severity (default 10)")
    ver.add_argument("--show-run", type=int, metavar="RUN_ID", help="print a stored historical run instead of verifying again")
    ver.add_argument("--json", action="store_true", help="print the full structured report as JSON")
    args = parser.parse_args(argv)
    if args.command == "verify" and not args.show_run and args.project_id is None and not args.project:
        parser.error("verify needs --project-id or --project (or --show-run RUN_ID)")

    try:
        if args.command == "eval-retrieval":
            return _eval_retrieval(args)
        if args.command == "ask":
            return _ask(args)
        if args.command == "verify":
            return _verify(args)
        ctx = bootstrap(Settings.from_env())
    except TrustDocError as exc:
        print(f"error: {exc}", file=sys.stderr if args.command in ("ask", "verify") else sys.stdout)
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
