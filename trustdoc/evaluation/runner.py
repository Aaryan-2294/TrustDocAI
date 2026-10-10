"""Run the four retrieval pipelines over a labelled dataset and report the results.

Pipelines:  bm25  |  faiss  |  hybrid_rrf (FAISS + BM25 + RRF)  |  hybrid_rerank (+ Cross-Encoder)

Metrics are computed on the ranked list BEFORE the relevance threshold (the
threshold is an abstention mechanism, not a ranking one). Nothing in this module
invents numbers: every figure comes from running the retriever, and the report
records whether stand-in (lexical stub) models were used so results cannot be
mistaken for real-model performance.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.evaluation.dataset import (
    DEFAULT_QUERIES,
    EvalDataset,
    EvaluationError,
    load_dataset,
    resolve_relevance,
)
from trustdoc.evaluation.metrics import first_relevant_rank, mean, recall_at_k, reciprocal_rank
from trustdoc.ingestion import IngestionService
from trustdoc.logging_setup import get_logger
from trustdoc.models import Project
from trustdoc.retrieval.embedding import Embedder
from trustdoc.retrieval.index_manager import IndexManager
from trustdoc.retrieval.reranker import Reranker
from trustdoc.retrieval.retriever import Retriever
from trustdoc.retrieval.types import RetrievalMode
from trustdoc.utils_time import utc_now_iso

log = get_logger("evaluation")

PIPELINES: dict[str, RetrievalMode] = {
    "bm25": RetrievalMode.BM25,
    "faiss": RetrievalMode.FAISS,
    "hybrid_rrf": RetrievalMode.HYBRID,
    "hybrid_rerank": RetrievalMode.HYBRID_RERANK,
}
PIPELINE_LABELS = {
    "bm25": "A. BM25 only",
    "faiss": "B. FAISS only",
    "hybrid_rrf": "C. Hybrid (FAISS + BM25 + RRF)",
    "hybrid_rerank": "D. Hybrid + Cross-Encoder rerank",
}
DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class QueryOutcome:
    query_id: str
    category: str
    text: str
    first_relevant_rank: int | None
    recall: dict[int, float]
    reciprocal_rank: float
    latency_ms: float
    retrieved: list[dict[str, Any]]


@dataclass
class PipelineResult:
    name: str
    outcomes: list[QueryOutcome] = field(default_factory=list)
    overall: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    meta: dict[str, Any]
    pipelines: dict[str, PipelineResult]


def _aggregate(outcomes: Sequence[QueryOutcome], ks: Sequence[int]) -> dict[str, float]:
    agg = {f"recall@{k}": mean([o.recall[k] for o in outcomes]) for k in ks}
    agg["mrr"] = mean([o.reciprocal_rank for o in outcomes])
    agg["mean_latency_ms"] = mean([o.latency_ms for o in outcomes])
    agg["n_queries"] = len(outcomes)
    return agg


def evaluate(
    db: Database,
    retriever: Retriever,
    project_id: int,
    dataset: EvalDataset,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    pipelines: Sequence[str] | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    warmup: bool = True,
) -> EvaluationReport:
    ks = tuple(sorted(set(ks)))
    names = list(pipelines or PIPELINES)
    unknown = [n for n in names if n not in PIPELINES]
    if unknown:
        raise EvaluationError(f"unknown pipelines: {unknown}")

    with db.session() as s:
        chunks = s.chunks.list_for_project(project_id)
        sources = s.sources.list_for_project(project_id)
    source_names = {src.id: src.name for src in sources}
    relevance = resolve_relevance(dataset, chunks, source_names)
    retriever.index_manager.ensure(project_id)

    depth = max(ks)
    results: dict[str, PipelineResult] = {}
    for name in names:
        mode = PIPELINES[name]
        if warmup:  # untimed: loads models / warms caches so latency reflects steady state
            retriever.search(project_id, dataset.queries[0].text, top_k=depth, mode=mode,
                             apply_threshold=False)
        outcome_list: list[QueryOutcome] = []
        for q in dataset.queries:
            res = retriever.search(project_id, q.text, top_k=depth, mode=mode, apply_threshold=False)
            ids = [e.chunk_id for e in res.ranked]
            groups = relevance[q.id]
            relevant_ids = set().union(*groups)
            outcome_list.append(QueryOutcome(
                query_id=q.id, category=q.category, text=q.text,
                first_relevant_rank=first_relevant_rank(ids, groups),
                recall={k: recall_at_k(ids, groups, k) for k in ks},
                reciprocal_rank=reciprocal_rank(ids, groups),
                latency_ms=res.timings_ms.get("total_ms", 0.0),
                retrieved=[
                    {"rank": e.final_rank, "chunk_id": e.chunk_id, "citation": e.citation.label,
                     "relevant": e.chunk_id in relevant_ids}
                    for e in res.ranked
                ],
            ))
        pr = PipelineResult(name, outcome_list, _aggregate(outcome_list, ks))
        for cat in dataset.categories():
            pr.by_category[cat] = _aggregate([o for o in outcome_list if o.category == cat], ks)
        results[name] = pr

    stub = [getattr(embedder, "is_stub", None), getattr(reranker, "is_stub", None)]
    meta = {
        "generated_at": utc_now_iso(),
        "dataset": {"name": dataset.name, "description": dataset.description,
                    "illustrative": dataset.illustrative, "n_queries": len(dataset.queries),
                    "categories": dataset.categories()},
        "corpus": {"chunks": len(chunks), "sources": len(sources)},
        "ks": list(ks),
        "pipelines": {n: PIPELINE_LABELS[n] for n in names},
        "embedder": {"model": getattr(embedder, "model_name", None), "is_stub": stub[0]},
        "reranker": {"model": getattr(reranker, "model_name", None), "is_stub": stub[1]},
        "uses_stand_in_models": bool(stub[0] or stub[1]),
        "retrieval_config": {"candidate_k": retriever._cfg.candidate_k, "rrf_k": retriever._cfg.rrf_k},
        "environment": _environment(),
    }
    return EvaluationReport(meta=meta, pipelines=results)


def _environment() -> dict[str, str]:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for dist in ("faiss-cpu", "rank-bm25", "sentence-transformers", "numpy", "torch"):
        try:
            env[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[dist] = "not installed"
    return env


# ------------------------------------------------------------------ reporting
def _r(x: float) -> float:
    return round(float(x), 4)


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    pipelines: dict[str, Any] = {}
    for name, pr in report.pipelines.items():
        pipelines[name] = {
            "label": PIPELINE_LABELS[name],
            "overall": {k: _r(v) for k, v in pr.overall.items()},
            "by_category": {c: {k: _r(v) for k, v in m.items()} for c, m in pr.by_category.items()},
            "queries": [
                {"id": o.query_id, "category": o.category, "text": o.text,
                 "first_relevant_rank": o.first_relevant_rank,
                 "reciprocal_rank": _r(o.reciprocal_rank),
                 "recall": {str(k): _r(v) for k, v in o.recall.items()},
                 "latency_ms": _r(o.latency_ms), "retrieved": o.retrieved}
                for o in pr.outcomes
            ],
        }
    return {"meta": report.meta, "results": pipelines}


def render_summary(report: EvaluationReport) -> str:
    m = report.meta
    ks = m["ks"]
    lines = [f"# Retrieval evaluation: {m['dataset']['name']}", ""]
    lines += ["> **How to read this.**"]
    if m["dataset"]["illustrative"]:
        lines += [f"> The dataset is an **illustrative fixture** ({m['dataset']['n_queries']} queries over "
                  f"{m['corpus']['chunks']} chunks). Differences between pipelines are NOT statistically meaningful."]
    if m["uses_stand_in_models"]:
        lines += ["> This run used **lexical stand-in models** "
                  f"(embedder: `{m['embedder']['model']}`, reranker: `{m['reranker']['model']}`). "
                  "The numbers measure the pipeline plumbing with deterministic stubs and are "
                  "**NOT the performance of the real Sentence-Transformer / Cross-Encoder models**. "
                  "Re-run with the real models (see README) to measure those."]
    else:
        lines += [f"> Real models were used (embedder `{m['embedder']['model']}`, "
                  f"reranker `{m['reranker']['model']}`)."]
    lines += ["", f"Generated {m['generated_at']} | candidate_k={m['retrieval_config']['candidate_k']}, "
                  f"rrf_k={m['retrieval_config']['rrf_k']} | {m['environment']}", ""]

    head = ["Pipeline", *[f"Recall@{k}" for k in ks], "MRR", "Latency ms"]
    lines += ["## Overall", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, pr in report.pipelines.items():
        o = pr.overall
        row = [PIPELINE_LABELS[name], *[f"{o[f'recall@{k}']:.3f}" for k in ks],
               f"{o['mrr']:.3f}", f"{o['mean_latency_ms']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")

    names = list(report.pipelines)
    kk = 5 if 5 in ks else ks[-1]
    for title, key in (("MRR", "mrr"), (f"Recall@{kk}", f"recall@{kk}")):
        lines += ["", f"## {title} by query type", "",
                  "| Query type | n | " + " | ".join(PIPELINE_LABELS[n].split(". ")[0] for n in names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for cat in report.meta["dataset"]["categories"]:
            n = report.pipelines[names[0]].by_category[cat]["n_queries"]
            vals = [f"{report.pipelines[p].by_category[cat][key]:.3f}" for p in names]
            lines.append(f"| {cat} | {int(n)} | " + " | ".join(vals) + " |")

    lines += ["", "## Rank of the first relevant chunk per query (blank = not retrieved)", "",
              "| Query | Type | " + " | ".join(PIPELINE_LABELS[n].split(". ")[0] for n in names) + " |",
              "|---|---|" + "---|" * len(names)]
    first = report.pipelines[names[0]].outcomes
    for i, o in enumerate(first):
        ranks = [report.pipelines[p].outcomes[i].first_relevant_rank for p in names]
        lines.append(f"| `{o.query_id}` {o.text} | {o.category} | "
                     + " | ".join("" if r is None else str(r) for r in ranks) + " |")
    lines += ["", "Recall@K counts a relevant item as found if any of its chunks is in the top K; "
                  "MRR is the mean reciprocal rank of the first relevant chunk (0 if absent). "
                  "Metrics use the ranking before the relevance threshold.", ""]
    return "\n".join(lines)


def write_report(report: EvaluationReport, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "results.json", "csv": out / "per_query.csv", "summary": out / "summary.md"}
    paths["json"].write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    ks = report.meta["ks"]
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pipeline", "query_id", "category", "query", "first_relevant_rank",
                    "reciprocal_rank", *[f"recall@{k}" for k in ks], "latency_ms"])
        for name, pr in report.pipelines.items():
            for o in pr.outcomes:
                w.writerow([name, o.query_id, o.category, o.text,
                            "" if o.first_relevant_rank is None else o.first_relevant_rank,
                            _r(o.reciprocal_rank), *[_r(o.recall[k]) for k in ks], _r(o.latency_ms)])
    paths["summary"].write_text(render_summary(report), encoding="utf-8")
    return paths


# ------------------------------------------------------------ one-call runner
def run_sample_evaluation(
    out_dir: str | Path,
    embedder: Embedder,
    reranker: Reranker | None,
    *,
    queries_path: str | Path = DEFAULT_QUERIES,
    ks: Sequence[int] = DEFAULT_KS,
    pipelines: Sequence[str] | None = None,
) -> EvaluationReport:
    """Ingest the dataset's sample project into an isolated temp workspace, index it,
    evaluate, and write results.json / per_query.csv / summary.md into ``out_dir``.

    Your own database and data directory are never touched.
    """
    dataset = load_dataset(queries_path)
    names = list(pipelines or PIPELINES)
    if reranker is None and "hybrid_rerank" in names:
        names.remove("hybrid_rerank")
    with tempfile.TemporaryDirectory(prefix="trustdoc-eval-") as tmp:
        env = dict(os.environ)
        env.update({"TRUSTDOC_DATA_DIR": tmp, "TRUSTDOC_LOG_TO_FILE": "false"})
        for key in ("TRUSTDOC_DB_PATH",):
            env.pop(key, None)
        settings = Settings.from_env(env)
        settings.ensure_directories()
        db = Database(settings.storage.db_path)
        db.initialize()
        with db.session() as s:
            project_id = s.projects.create(Project(name=dataset.name)).id
        batch = IngestionService(db, settings).ingest_directory(project_id, dataset.project_dir)
        if batch.failures:
            detail = "; ".join(f"{f.name}: {f.error}" for f in batch.failures)
            raise EvaluationError(f"could not ingest the sample project ({detail})")
        retriever = Retriever(db, settings, IndexManager(db, settings, embedder), embedder, reranker)
        report = evaluate(db, retriever, project_id, dataset, ks=ks, pipelines=names,
                          embedder=embedder, reranker=reranker)
    write_report(report, out_dir)
    return report
