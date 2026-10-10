"""The unified retrieval facade.

    retriever.search(project_id, query, top_k=5) -> SearchResult

Callers (Ask, Verify) get ranked ``Evidence`` and never see FAISS, BM25, RRF,
the Cross-Encoder or index files. Pipeline for ``HYBRID_RERANK``::

    query -> BM25 top-N  ┐
                          ├-> RRF -> top-N candidates -> Cross-Encoder -> ranked
    query -> FAISS top-N ┘                                     |
                                              relevance threshold -> evidence / abstain

where N = max(candidate_k, top_k). The Cross-Encoder only ever scores the fused
candidates. Models and indexes are loaded once and reused across queries.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.exceptions import RetrievalError
from trustdoc.logging_setup import get_logger
from trustdoc.models import Chunk, Source
from trustdoc.retrieval.embedding import Embedder
from trustdoc.retrieval.fusion import reciprocal_rank_fusion
from trustdoc.retrieval.index_manager import IndexManager
from trustdoc.retrieval.reranker import Reranker
from trustdoc.retrieval.text import chunk_index_text
from trustdoc.retrieval.threshold import apply_relevance_threshold
from trustdoc.retrieval.types import (
    Citation,
    Evidence,
    FusedHit,
    RankedHit,
    RankedList,
    RetrievalMode,
    SearchResult,
)

log = get_logger("retrieval")


class Retriever:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        index_manager: IndexManager,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        auto_index: bool = True,
    ) -> None:
        self._db = db
        self._cfg = settings.retrieval
        self._indexes = index_manager
        self._embedder = embedder
        self._reranker = reranker
        self._auto_index = auto_index
        self._warned_threshold = False

    @property
    def index_manager(self) -> IndexManager:
        return self._indexes

    # ---------------------------------------------------------------- search
    def search(
        self,
        project_id: int,
        query: str,
        *,
        top_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID_RERANK,
        apply_threshold: bool = True,
        threshold: float | None = None,
    ) -> SearchResult:
        """Retrieve ranked evidence for ``query`` from one project's knowledge base.

        ``result.ranked`` is the top-k *before* the relevance threshold (useful for
        evaluation); ``result.evidence`` is what survives it; ``result.abstained``
        signals insufficient evidence. The threshold only applies to the
        reranked pipeline - other modes have no comparable score.
        """
        query = (query or "").strip()
        if not query:
            raise RetrievalError("query must not be empty")
        top_k = self._cfg.final_k if top_k is None else top_k
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        mode = RetrievalMode(mode)
        if mode == RetrievalMode.HYBRID_RERANK and self._reranker is None:
            raise RetrievalError("hybrid_rerank mode needs a reranker; none is configured")

        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000, 3)
            return now

        t = time.perf_counter()
        index = self._indexes.ensure(project_id) if self._auto_index else self._indexes.load(project_id)
        t = lap("index_ms", t)
        depth = max(self._cfg.candidate_k, top_k)

        bm25_hits: list[RankedHit] = []
        faiss_hits: list[RankedHit] = []
        if mode in (RetrievalMode.BM25, RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK):
            bm25_hits = index.bm25.search(query, depth)
            t = lap("bm25_ms", t)
        if mode in (RetrievalMode.FAISS, RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK):
            vector = self._embedder.embed_query(query)
            t = lap("embed_query_ms", t)
            faiss_hits = index.faiss.search(vector, depth)
            t = lap("faiss_ms", t)
        bm25_by = {h.chunk_id: h for h in bm25_hits}
        faiss_by = {h.chunk_id: h for h in faiss_hits}

        fused: list[FusedHit] = []
        if mode in (RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK):
            fused = reciprocal_rank_fusion(
                [RankedList("bm25", bm25_hits), RankedList("faiss", faiss_hits)], self._cfg.rrf_k
            )
            t = lap("rrf_ms", t)
        fused_by = {f.chunk_id: f for f in fused}

        # ---- final ordering of chunk ids for this mode
        rerank_scores: dict[int, float] = {}
        rerank_ranks: dict[int, int] = {}
        if mode == RetrievalMode.BM25:
            order = [h.chunk_id for h in bm25_hits]
        elif mode == RetrievalMode.FAISS:
            order = [h.chunk_id for h in faiss_hits]
        elif mode == RetrievalMode.HYBRID:
            order = [f.chunk_id for f in fused]
        else:
            pool = fused[:depth]
            pool_ids = [f.chunk_id for f in pool]
            loaded = self._hydrate(project_id, pool_ids)
            t = lap("hydrate_ms", t)
            pool = [f for f in pool if f.chunk_id in loaded]
            scores = self._reranker.score(  # type: ignore[union-attr]
                query, [chunk_index_text(loaded[f.chunk_id][0]) for f in pool]
            )
            if len(scores) != len(pool):
                raise RetrievalError("reranker returned a different number of scores than candidates")
            clean = [float("-inf") if (s is None or (isinstance(s, float) and math.isnan(s))) else float(s)
                     for s in scores]
            ranked_pool = sorted(zip(pool, clean), key=lambda p: (-p[1], p[0].rank, p[0].chunk_id))
            for position, (f, s) in enumerate(ranked_pool, start=1):
                rerank_scores[f.chunk_id] = s
                rerank_ranks[f.chunk_id] = position
            order = [f.chunk_id for f, _ in ranked_pool]
            t = lap("rerank_ms", t)

        order = order[:top_k]
        if mode != RetrievalMode.HYBRID_RERANK:
            loaded = self._hydrate(project_id, order)
            t = lap("hydrate_ms", t)

        ranked: list[Evidence] = []
        for cid in order:
            if cid not in loaded:       # chunk deleted since the index was built
                continue
            chunk, source = loaded[cid]
            ev = self._evidence(chunk, source, mode)
            if cid in bm25_by:
                ev.bm25_rank, ev.bm25_score = bm25_by[cid].rank, bm25_by[cid].score
            if cid in faiss_by:
                ev.faiss_rank, ev.faiss_score = faiss_by[cid].rank, faiss_by[cid].score
            if cid in fused_by:
                ev.rrf_rank, ev.rrf_score = fused_by[cid].rank, fused_by[cid].rrf_score
            if cid in rerank_scores:
                ev.rerank_score, ev.rerank_rank = rerank_scores[cid], rerank_ranks[cid]
            ranked.append(ev)
        for position, ev in enumerate(ranked, start=1):
            ev.final_rank = position

        # ---- relevance threshold (reranked pipeline only)
        result = SearchResult(
            query=query, project_id=project_id, mode=mode, ranked=ranked, evidence=list(ranked),
            abstained=not ranked, candidate_count=len(rerank_scores) or len(order),
            index_fingerprint=index.manifest.knowledge_hash,
        )
        if rerank_scores:
            result.best_score = max(rerank_scores.values())
        if mode == RetrievalMode.HYBRID_RERANK and apply_threshold:
            thr = self._cfg.relevance_threshold if threshold is None else threshold
            if thr == 0.0 and not self._warned_threshold:
                log.warning(
                    "relevance_threshold is 0.0, the uncalibrated placeholder default; "
                    "calibrate it experimentally for your reranker before trusting abstention"
                )
                self._warned_threshold = True
            decision = apply_relevance_threshold(ranked, thr, lambda e: e.rerank_score)
            result.threshold, result.threshold_applied = thr, True
            result.evidence, result.abstained = decision.kept, decision.abstain
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
        result.timings_ms = timings
        return result

    # --------------------------------------------------------------- helpers
    def _hydrate(self, project_id: int, chunk_ids: Sequence[int]) -> dict[int, tuple[Chunk, Source]]:
        with self._db.session() as s:
            chunks = s.chunks.get_many(list(chunk_ids))
            sources: dict[int, Source] = {}
            out: dict[int, tuple[Chunk, Source]] = {}
            for chunk in chunks:
                if chunk.project_id != project_id:   # never leak another project's chunk
                    continue
                if chunk.source_id not in sources:
                    src = s.sources.get(chunk.source_id)
                    if src is None:
                        continue
                    sources[chunk.source_id] = src
                out[chunk.id] = (chunk, sources[chunk.source_id])
        return out

    @staticmethod
    def _evidence(chunk: Chunk, source: Source, mode: RetrievalMode) -> Evidence:
        # Only the logical source name feeds the citation; stored_path is never used here.
        citation = Citation(
            source_id=source.id, source_name=source.name, source_type=source.source_type.value,
            label=chunk.location_label(source.name), chunk_id=chunk.id,
            start_line=chunk.start_line, end_line=chunk.end_line, page_number=chunk.page_number,
            heading=chunk.heading, qualified_name=chunk.metadata.get("qualified_name"),
        )
        return Evidence(
            chunk_id=chunk.id, source_id=source.id, project_id=chunk.project_id, text=chunk.text,
            chunk_type=chunk.chunk_type.value, source_name=source.name,
            source_type=source.source_type.value, citation=citation, method=mode,
            entity_id=chunk.entity_id, metadata=dict(chunk.metadata),
        )
