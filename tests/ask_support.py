"""Mocks for the Ask tests: no numpy, no models, no network."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from trustdoc.llm.base import GenerationResult, LLMBackend
from trustdoc.retrieval.types import Citation, Evidence, RetrievalMode, SearchResult


def make_evidence(
    chunk_id: int = 1,
    text: str = "def compute_backoff(attempt): ...",
    *,
    project_id: int = 1,
    source_name: str = "pkg/module.py",
    source_type: str = "python",
    start: int | None = 10,
    end: int | None = 24,
    page: int | None = None,
    heading: str | None = None,
    qualified_name: str | None = None,
    label: str | None = None,
    rerank: float | None = 0.9,
    source_id: int | None = None,
) -> Evidence:
    if label is None:
        if page is not None:
            label = f"{source_name}, p.{page}"
        elif heading:
            label = f"{source_name}:{start}-{end} \u00a7 {heading}"
        else:
            label = f"{source_name}:{start}-{end}"
    sid = source_id if source_id is not None else chunk_id
    citation = Citation(
        source_id=sid, source_name=source_name, source_type=source_type, label=label,
        chunk_id=chunk_id, start_line=start, end_line=end, page_number=page, heading=heading,
        qualified_name=qualified_name,
    )
    return Evidence(
        chunk_id=chunk_id, source_id=sid, project_id=project_id, text=text,
        chunk_type="code" if source_type == "python" else source_type, source_name=source_name,
        source_type=source_type, citation=citation, method=RetrievalMode.HYBRID_RERANK,
        rerank_score=rerank, final_rank=chunk_id,
    )


def make_search_result(
    evidence: list[Evidence],
    *,
    project_id: int = 1,
    abstained: bool = False,
    ranked: list[Evidence] | None = None,
    threshold: float | None = 0.5,
    threshold_applied: bool = True,
) -> SearchResult:
    ranked = evidence if ranked is None else ranked
    return SearchResult(
        query="q", project_id=project_id, mode=RetrievalMode.HYBRID_RERANK, ranked=ranked,
        evidence=evidence, abstained=abstained, threshold=threshold,
        threshold_applied=threshold_applied,
        best_score=max((e.rerank_score for e in ranked if e.rerank_score is not None), default=None),
    )


class FakeRetriever:
    """Returns a prepared SearchResult (or raises) and records every call."""

    def __init__(self, result: SearchResult | Exception) -> None:
        self.result = result
        self.calls: list[dict] = []

    def search(self, project_id, query, *, top_k=None, **kwargs):
        self.calls.append({"project_id": project_id, "query": query, "top_k": top_k})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class ScriptedLLM(LLMBackend):
    """Returns canned text (or raises) and records prompts."""

    name = "scripted"
    is_stub = False

    def __init__(self, reply: str | Exception = "Answer [E1].", *, finish_reason: str | None = "stop") -> None:
        super().__init__("scripted-model")
        self.reply = reply
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def generate(self, prompt, system_prompt=None, *, max_tokens=None, temperature=None):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if isinstance(self.reply, Exception):
            raise self.reply
        return GenerationResult(
            text=self.reply, model=self.model_name, backend=self.name, finish_reason=self.finish_reason,
        )


# ---- a local HTTP server mimicking Ollama's documented /api/chat contract (NOT a real Ollama)
class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        srv.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
        try:
            if srv.delay:
                time.sleep(srv.delay)
            self.send_response(srv.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = srv.payload
            self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class FakeOllama:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.requests, self.server.status, self.server.delay = [], 200, 0
        self.respond({"model": "m", "message": {"role": "assistant", "content": "Hello [E1]"}, "done": True,
                      "done_reason": "stop", "prompt_eval_count": 11, "eval_count": 4})
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def respond(self, payload, status=200, delay=0):
        self.server.payload, self.server.status, self.server.delay = payload, status, delay

    @property
    def requests(self):
        return self.server.requests
