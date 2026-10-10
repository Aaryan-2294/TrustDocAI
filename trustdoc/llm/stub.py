"""Deterministic extractive stand-in. NOT a language model.

It quotes the beginning of the first evidence block it finds in the prompt and
cites it. It exists so the Ask pipeline can be smoke-tested and demonstrated
with no model installed; its output says nothing about real LLM behaviour.
"""

from __future__ import annotations

import re
import time

from trustdoc.llm.base import GenerationResult, LLMBackend


class ExtractiveStubLLM(LLMBackend):
    name = "stub"
    is_stub = True

    def __init__(self, model_name: str = "extractive-stub") -> None:
        super().__init__(model_name or "extractive-stub")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        from trustdoc.workflows.prompts import EVIDENCE_BLOCK_RE  # local import: llm must not need workflows at import time

        started = time.perf_counter()
        match = EVIDENCE_BLOCK_RE.search(prompt)
        if match:
            snippet = re.sub(r"\s+", " ", match.group("text")).strip()[:240]
            text = (f"Stand-in response (not a language model). The top retrieved evidence "
                    f"[{match.group('id')}] begins: \"{snippet}\"")
        else:
            text = "Stand-in response (not a language model): no evidence was supplied."
        return GenerationResult(
            text=text, model=self.model_name, backend=self.name, finish_reason="stop",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
