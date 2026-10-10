"""Optional in-process backend using Hugging Face ``transformers`` (CPU by default).

``transformers``/``torch`` are imported lazily on first ``generate`` call, and the
model is loaded once and then reused. Nothing is imported or downloaded at
construction time. If the model is not in the local Hugging Face cache it is
downloaded on first use (set ``HF_HUB_OFFLINE=1`` to forbid that).

Resource notes: weights load in the library's default precision (float32 on
CPU), i.e. roughly 4 bytes per parameter of RAM, plus activations; a 1.5B-parameter
model needs on the order of 6+ GB. CPU generation is slow (seconds per token for
larger models). ``timeout_seconds`` maps to ``max_time``, which stops generation
cooperatively and returns the partial text with ``finish_reason="time_limit"``.
"""

from __future__ import annotations

import threading
import time

from trustdoc.exceptions import (
    LLMEmptyOutputError,
    LLMError,
    LLMNotConfiguredError,
    LLMResponseError,
    LLMUnavailableError,
    MissingDependencyError,
)
from trustdoc.llm.base import GenerationResult, LLMBackend
from trustdoc.logging_setup import get_logger

log = get_logger("llm.transformers")


class TransformersBackend(LLMBackend):
    name = "transformers"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_tokens: int = 4096,
        timeout_seconds: float = 120,
    ) -> None:
        if not model or not model.strip():
            raise LLMNotConfiguredError(
                "The transformers backend needs a Hugging Face model id. Set TRUSTDOC_LLM_MODEL "
                "(an instruction-tuned chat model that fits your RAM)."
            )
        super().__init__(model.strip())
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_tokens = context_tokens
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._loaded: tuple | None = None

    def _load(self):
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            try:
                import torch
                import transformers
            except ImportError as exc:
                raise MissingDependencyError(
                    "The transformers LLM backend needs torch and transformers. Install them with "
                    '`pip install -e ".[retrieval]"` (or `pip install torch transformers`).'
                ) from exc
            log.warning(
                "Loading language model %r (downloaded on first use if not cached; CPU inference is slow)",
                self.model_name,
            )
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
                model.eval()
            except OSError as exc:
                raise LLMUnavailableError(
                    f"Could not load model {self.model_name!r}: {exc}. Check the model id and your "
                    "network connection, or pre-download the model."
                ) from exc
            self._loaded = (torch, tokenizer, model)
            return self._loaded

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        torch, tokenizer, model = self._load()
        max_new = self._max_tokens if max_tokens is None else max_tokens
        temp = self._temperature if temperature is None else temperature

        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages.append({"role": "user", "content": prompt})
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, AttributeError, TypeError):
            log.warning("Model %r has no chat template; using a plain-text prompt", self.model_name)
            chat_text = (f"System: {system_prompt}\n\n" if system_prompt else "") + \
                f"User: {prompt}\n\nAssistant:"

        inputs = tokenizer(chat_text, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[1])
        if prompt_tokens + max_new > self._context_tokens:
            raise LLMResponseError(
                f"Prompt ({prompt_tokens} tokens) plus max_tokens ({max_new}) exceeds the context "
                f"window ({self._context_tokens}). Lower TRUSTDOC_ASK_MAX_EVIDENCE_CHARS / "
                "TRUSTDOC_ASK_MAX_EVIDENCE_ITEMS or raise TRUSTDOC_LLM_CONTEXT_TOKENS."
            )

        kwargs = {
            "max_new_tokens": max_new,
            "do_sample": temp > 0,
            "max_time": float(self._timeout),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }
        if temp > 0:
            kwargs["temperature"] = temp

        started = time.perf_counter()
        try:
            with torch.no_grad():
                output = model.generate(**inputs, **kwargs)
        except RuntimeError as exc:  # e.g. out of memory
            raise LLMError(f"Generation failed for {self.model_name!r}: {exc}") from exc
        elapsed = time.perf_counter() - started

        new_ids = output[0][prompt_tokens:]
        completion_tokens = len(new_ids)
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not text:
            raise LLMEmptyOutputError(f"Model {self.model_name!r} returned an empty response")
        if elapsed >= self._timeout:
            finish = "time_limit"
        elif completion_tokens >= max_new:
            finish = "length"
        else:
            finish = "stop"
        return GenerationResult(
            text=text, model=self.model_name, backend=self.name, finish_reason=finish,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            elapsed_ms=round(elapsed * 1000, 1),
        )
