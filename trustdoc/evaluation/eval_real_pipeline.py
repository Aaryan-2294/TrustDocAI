"""Reproducible evaluation runner for TrustDoc AI's real-model ASK and retrieval pipeline.

Runs against Project 1 (retry-task-queue) using:
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: Ollama qwen2.5-coder:1.5b

Run standalone via:
    python trustdoc/evaluation/eval_real_pipeline.py
"""

from __future__ import annotations

import os
import sys

# Ensure Hugging Face cache points to E: drive
os.environ["HF_HOME"] = r"E:\AI\Cache\HuggingFace"
os.environ["HF_HUB_CACHE"] = r"E:\AI\Cache\HuggingFace\hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trustdoc.app import bootstrap
from trustdoc.config import Settings
from trustdoc.retrieval.factory import create_retriever
from trustdoc.workflows.ask import create_ask_service
from trustdoc.workflows.prompts import build_prompt, SYSTEM_PROMPT
from trustdoc.workflows.types import AskStatus, GroundingLevel

DATASET_PATH = Path(__file__).parent / "data" / "retry_task_queue_eval.json"
OUTPUT_DIR = Path(__file__).parent / "reports"


def resolve_gold_chunks(db, project_id: int, relevant_refs: list[dict[str, Any]]) -> set[int]:
    """Map human-labeled relevant references to actual database chunk IDs."""
    if not relevant_refs:
        return set()

    gold_chunk_ids: set[int] = set()
    with db.session() as s:
        chunks = s.chunks.list_for_project(project_id)
        sources = {src.id: src.name for src in s.sources.list_for_project(project_id)}

        for ref in relevant_refs:
            ref_source = ref.get("source")
            ref_qn = ref.get("qualified_name")
            ref_heading = ref.get("heading")

            for c in chunks:
                src_name = sources.get(c.source_id)
                if src_name != ref_source:
                    continue

                if ref_qn is not None:
                    if c.metadata.get("qualified_name") == ref_qn:
                        gold_chunk_ids.add(c.id)
                elif ref_heading is not None:
                    if c.heading == ref_heading:
                        gold_chunk_ids.add(c.id)
                else:
                    gold_chunk_ids.add(c.id)

    return gold_chunk_ids


def evaluate_pipeline(
    dataset_path: Path = DATASET_PATH,
    project_id: int = 1,
    candidate_thresholds: list[float] | None = None,
    output_name: str = "eval_results_pass5",
) -> dict[str, Any]:
    if candidate_thresholds is None:
        candidate_thresholds = [-15.0, -10.0, -5.0, -2.0, -1.0, 0.0]

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries = raw["queries"]

    settings = Settings.from_env()
    ctx = bootstrap(settings)
    retriever = create_retriever(ctx.db, ctx.settings)
    ask_service = create_ask_service(ctx.db, ctx.settings, retriever=retriever)

    print(f"Loaded dataset: {raw['name']} ({len(queries)} queries)")
    print(f"Project ID: {project_id}")
    print(f"LLM Provider: {settings.llm.provider} | Model: {settings.llm.model}")
    print(f"Configured Global Relevance Threshold: {settings.retrieval.relevance_threshold}")
    print(f"Configured Ask Relevance Threshold: {settings.ask.relevance_threshold}")

    results = []
    gold_scores: list[float] = []
    negative_scores: list[float] = []

    # Map all gold chunk IDs first
    for q in queries:
        q["gold_chunk_ids"] = list(resolve_gold_chunks(ctx.db, project_id, q.get("relevant", [])))

    print("\n" + "=" * 70)
    print("RUNNING END-TO-END PIPELINE EVALUATION")
    print("=" * 70)

    for idx, q in enumerate(queries, start=1):
        qid = q["id"]
        text = q["text"]
        is_answerable = q["is_answerable"]
        gold_ids = set(q["gold_chunk_ids"])

        print(f"\n[{idx:02d}/{len(queries)}] {qid} ({q['category']})")
        print(f"  Q: {text}")

        # 1. Retrieval
        t0 = time.perf_counter()
        search_res = retriever.search(project_id, text, top_k=5, apply_threshold=True)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        ranked_ids = [e.chunk_id for e in search_res.ranked]
        evidence_ids = [e.chunk_id for e in search_res.evidence]

        # Record scores
        for e in search_res.ranked:
            if e.rerank_score is not None:
                if e.chunk_id in gold_ids:
                    gold_scores.append(e.rerank_score)
                else:
                    negative_scores.append(e.rerank_score)

        # Retrieval Metrics
        recall_at_1 = 0.0
        recall_at_3 = 0.0
        recall_at_5 = 0.0
        rr = 0.0

        if gold_ids:
            hits_1 = set(ranked_ids[:1]) & gold_ids
            hits_3 = set(ranked_ids[:3]) & gold_ids
            hits_5 = set(ranked_ids[:5]) & gold_ids

            recall_at_1 = 1.0 if hits_1 else 0.0
            recall_at_3 = 1.0 if hits_3 else 0.0
            recall_at_5 = 1.0 if hits_5 else 0.0

            for rank_pos, cid in enumerate(ranked_ids, start=1):
                if cid in gold_ids:
                    rr = 1.0 / rank_pos
                    break

        # 2. ASK workflow execution
        t_ask0 = time.perf_counter()
        ask_res = ask_service.ask(project_id, text, top_k=5)
        total_ask_ms = (time.perf_counter() - t_ask0) * 1000

        # Citations check
        citations_list = [c.to_dict() for c in ask_res.citations]
        cited_chunk_ids = [c.chunk_id for c in ask_res.citations]
        invalid_cids = list(ask_res.invalid_citation_ids)
        is_citation_valid = len(invalid_cids) == 0

        # Outcome Classification
        status = ask_res.status.value
        grounding = ask_res.grounding.value if ask_res.grounding else "none"

        is_false_abstention = False
        is_true_abstention = False
        is_hallucination = False
        is_supported_answer = False

        if is_answerable:
            if ask_res.status == AskStatus.INSUFFICIENT_EVIDENCE:
                is_false_abstention = True
            elif ask_res.status == AskStatus.ANSWERED:
                # Check if cited evidence has gold chunks
                if any(cid in gold_ids for cid in cited_chunk_ids):
                    is_supported_answer = True
                else:
                    # answered without gold chunk citation
                    is_supported_answer = False
        else:
            # unanswerable or false premise
            if ask_res.status == AskStatus.INSUFFICIENT_EVIDENCE:
                is_true_abstention = True
            elif ask_res.status == AskStatus.ANSWERED:
                is_hallucination = True

        print(f"  Status: {status} | Grounding: {grounding} | Reason: {ask_res.abstention_reason} | Time: {total_ask_ms:.1f}ms")
        if ask_res.citations:
            print(f"  Citations: {[c.label for c in ask_res.citations]}")
        if is_false_abstention:
            print(f"  [!] False Abstention: Model abstained on answerable query.")
            print(f"      Model text: {ask_res.answer[:120]}...")
        elif is_supported_answer:
            print(f"  [OK] Supported Answer: {ask_res.answer[:120]}...")
        elif is_true_abstention:
            print(f"  [OK] True Abstention on unanswerable query.")
        elif is_hallucination:
            print(f"  [!] Hallucination on unanswerable query: {ask_res.answer[:120]}...")

        # Threshold behavior for this query
        threshold_outcomes = {}
        for thr in candidate_thresholds:
            survived = [e for e in search_res.ranked if (e.rerank_score or -999.0) >= thr]
            threshold_outcomes[str(thr)] = {
                "survived_count": len(survived),
                "has_gold": any(e.chunk_id in gold_ids for e in survived) if gold_ids else False,
                "abstained": len(survived) == 0,
            }

        q_record = {
            "id": qid,
            "category": q["category"],
            "text": text,
            "is_answerable": is_answerable,
            "gold_chunk_ids": list(gold_ids),
            "retrieval": {
                "retrieval_ms": retrieval_ms,
                "ranked_chunk_ids": ranked_ids,
                "ranked_scores": [e.rerank_score for e in search_res.ranked],
                "evidence_chunk_ids": evidence_ids,
                "recall_at_1": recall_at_1,
                "recall_at_3": recall_at_3,
                "recall_at_5": recall_at_5,
                "reciprocal_rank": rr,
            },
            "ask": {
                "total_ms": total_ask_ms,
                "status": status,
                "grounding": grounding,
                "backend": ask_res.backend,
                "model_name": ask_res.model_name,
                "abstained": ask_res.abstained,
                "abstention_reason": ask_res.abstention_reason,
                "answer": ask_res.answer,
                "raw_answer": ask_res.raw_answer,
                "citations": citations_list,
                "invalid_citation_ids": invalid_cids,
                "is_citation_valid": is_citation_valid,
                "is_supported_answer": is_supported_answer,
                "is_false_abstention": is_false_abstention,
                "is_true_abstention": is_true_abstention,
                "is_hallucination": is_hallucination,
            },
            "threshold_outcomes": threshold_outcomes,
        }
        results.append(q_record)

    # ------------------------------------------------------------- Summary Metrics
    answerable_queries = [r for r in results if r["is_answerable"]]
    unanswerable_queries = [r for r in results if not r["is_answerable"]]

    mean_recall_1 = sum(r["retrieval"]["recall_at_1"] for r in answerable_queries) / len(answerable_queries)
    mean_recall_3 = sum(r["retrieval"]["recall_at_3"] for r in answerable_queries) / len(answerable_queries)
    mean_recall_5 = sum(r["retrieval"]["recall_at_5"] for r in answerable_queries) / len(answerable_queries)
    mean_mrr = sum(r["retrieval"]["reciprocal_rank"] for r in answerable_queries) / len(answerable_queries)

    supported_count = sum(1 for r in answerable_queries if r["ask"]["is_supported_answer"])
    deterministic_count = sum(1 for r in answerable_queries if r["ask"]["is_supported_answer"] and r["ask"].get("backend") == "deterministic")
    llm_supported_count = sum(1 for r in answerable_queries if r["ask"]["is_supported_answer"] and r["ask"].get("backend") != "deterministic")
    false_abstention_count = sum(1 for r in answerable_queries if r["ask"]["is_false_abstention"])

    true_abstention_count = sum(1 for r in unanswerable_queries if r["ask"]["is_true_abstention"])
    hallucination_count = sum(1 for r in unanswerable_queries if r["ask"]["is_hallucination"])
    retrieval_filtered_unans = sum(1 for r in unanswerable_queries if r["ask"].get("abstention_reason") == "below_relevance_threshold")
    retrieval_filtered_total = sum(1 for r in results if r["ask"].get("abstention_reason") == "below_relevance_threshold")

    all_citations = sum(len(r["ask"]["citations"]) for r in results)
    invalid_citations = sum(len(r["ask"]["invalid_citation_ids"]) for r in results)
    citation_validity_pct = 100.0 if (all_citations + invalid_citations) == 0 else (all_citations / (all_citations + invalid_citations) * 100.0)

    avg_gold_score = sum(gold_scores) / len(gold_scores) if gold_scores else 0.0
    avg_neg_score = sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    min_gold_score = min(gold_scores) if gold_scores else 0.0
    max_gold_score = max(gold_scores) if gold_scores else 0.0
    min_neg_score = min(negative_scores) if negative_scores else 0.0
    max_neg_score = max(negative_scores) if negative_scores else 0.0

    # Threshold trade-off evaluation
    threshold_tradeoff = {}
    for thr in candidate_thresholds:
        k_str = str(thr)
        # On answerable queries: true positive = query retained gold chunk; false negative = query abstained or dropped gold
        retained_gold = sum(1 for r in answerable_queries if r["threshold_outcomes"][k_str]["has_gold"])
        # On unanswerable queries: true negative = query abstained; false positive = query survived
        abstained_unans = sum(1 for r in unanswerable_queries if r["threshold_outcomes"][k_str]["abstained"])
        false_positives = sum(1 for r in unanswerable_queries if not r["threshold_outcomes"][k_str]["abstained"])

        ans_recall = retained_gold / len(answerable_queries) if answerable_queries else 0.0
        unans_abstention_rate = abstained_unans / len(unanswerable_queries) if unanswerable_queries else 0.0

        threshold_tradeoff[k_str] = {
            "threshold": thr,
            "answerable_gold_retained_pct": round(ans_recall * 100, 1),
            "unanswerable_abstention_pct": round(unans_abstention_rate * 100, 1),
            "false_positives": false_positives,
            "false_negatives": len(answerable_queries) - retained_gold,
        }

    summary = {
        "dataset_name": raw["name"],
        "total_queries": len(queries),
        "answerable_count": len(answerable_queries),
        "unanswerable_count": len(unanswerable_queries),
        "retrieval": {
            "recall_at_1": round(mean_recall_1 * 100, 1),
            "recall_at_3": round(mean_recall_3 * 100, 1),
            "recall_at_5": round(mean_recall_5 * 100, 1),
            "mrr": round(mean_mrr, 3),
        },
        "score_distribution": {
            "gold_scores": {
                "count": len(gold_scores),
                "mean": round(avg_gold_score, 2),
                "min": round(min_gold_score, 2),
                "max": round(max_gold_score, 2),
            },
            "negative_scores": {
                "count": len(negative_scores),
                "mean": round(avg_neg_score, 2),
                "min": round(min_neg_score, 2),
                "max": round(max_neg_score, 2),
            },
        },
        "end_to_end_ask": {
            "answerable_supported_count": supported_count,
            "answerable_supported_pct": round(supported_count / len(answerable_queries) * 100, 1),
            "answerable_deterministic_count": deterministic_count,
            "answerable_llm_count": llm_supported_count,
            "false_abstentions_count": false_abstention_count,
            "false_abstentions_pct": round(false_abstention_count / len(answerable_queries) * 100, 1),
            "unanswerable_true_abstentions_count": true_abstention_count,
            "unanswerable_true_abstentions_pct": round(true_abstention_count / len(unanswerable_queries) * 100, 1),
            "unanswerable_hallucinations_count": hallucination_count,
            "retrieval_filtered_unanswerable_count": retrieval_filtered_unans,
            "retrieval_filtered_total_count": retrieval_filtered_total,
            "citation_validity_pct": round(citation_validity_pct, 1),
        },
        "threshold_tradeoff": threshold_tradeoff,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "eval_results.json"
    out_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    cf_file = OUTPUT_DIR / f"{output_name}.json"
    cf_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote full evaluation results to {out_file} and {cf_file}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    d_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_PATH
    out_name = sys.argv[2] if len(sys.argv) > 2 else "eval_results_pass5"
    evaluate_pipeline(dataset_path=d_path, output_name=out_name)
