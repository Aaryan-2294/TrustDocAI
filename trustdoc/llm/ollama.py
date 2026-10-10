"""Local Ollama backend (https://ollama.com) over its HTTP chat API.

Uses only the standard library (``urllib``), so it adds no Python dependency.
Ollama runs the model as a separate local server process; install it, start it
(``ollama serve``, or the desktop app) and fetch a model with ``ollama pull``.
Nothing is downloaded by TrustDoc itself.

Notes
* ``stream=false``: one response, so ``timeout_seconds`` bounds how long we wait
  for the whole completion (CPU generation can be slow; raise it if needed).
* Environment proxies are deliberately ignored: this is a local-server call.
* ``options.num_ctx`` is set explicitly because servers may otherwise silently
  truncate long prompts to a small default context window.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from trustdoc.exceptions import (
    LLMEmptyOutputError,
    LLMNotConfiguredError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from trustdoc.llm.base import GenerationResult, LLMBackend
from trustdoc.logging_setup import get_logger

log = get_logger("llm.ollama")

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_tokens: int = 4096,
        timeout_seconds: float = 120,
    ) -> None:
        if not model or not model.strip():
            raise LLMNotConfiguredError(
                "The ollama backend needs a model name. Set TRUSTDOC_LLM_MODEL "
                "(e.g. qwen2.5:1.5b) and run `ollama pull <model>` first."
            )
        super().__init__(model.strip())
        self.base_url = _validate_base_url(base_url or DEFAULT_BASE_URL)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_tokens = context_tokens
        self._timeout = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": self._max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self._context_tokens,
            },
        }
        started = time.perf_counter()
        data = self._post("/api/chat", payload)
        elapsed = (time.perf_counter() - started) * 1000

        if data.get("error"):
            raise LLMResponseError(f"Ollama reported an error: {data['error']}")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama response has no message content")
        text = content.strip()
        if not text:
            raise LLMEmptyOutputError(f"Model {self.model_name!r} returned an empty response")
        return GenerationResult(
            text=text,
            model=str(data.get("model") or self.model_name),
            backend=self.name,
            finish_reason=data.get("done_reason"),
            prompt_tokens=_int_or_none(data.get("prompt_eval_count")),
            completion_tokens=_int_or_none(data.get("eval_count")),
            elapsed_ms=round(elapsed, 1),
        )

    # ------------------------------------------------------------------ HTTP
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            if exc.code == 404:
                raise LLMUnavailableError(
                    f"Ollama does not have model {self.model_name!r} ({detail}). "
                    f"Download it with: ollama pull {self.model_name}"
                ) from exc
            raise LLMResponseError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise self._timeout_error() from exc
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url} ({exc.reason}). Install Ollama and start "
                "it (`ollama serve`), or change TRUSTDOC_LLM_BASE_URL."
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise self._timeout_error() from exc
        except (http.client.HTTPException, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(f"Connection to Ollama at {self.base_url} failed: {exc}") from exc

        try:
            data = json.loads(body)
        except ValueError as exc:
            raise LLMResponseError("Ollama returned a response that is not valid JSON") from exc
        if not isinstance(data, dict):
            raise LLMResponseError("Ollama returned an unexpected response shape")
        return data

    def _timeout_error(self) -> LLMTimeoutError:
        return LLMTimeoutError(
            f"Ollama did not answer within {self._timeout:g}s. CPU generation can be slow; "
            "raise TRUSTDOC_LLM_TIMEOUT_SECONDS or use a smaller model."
        )


def _validate_base_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LLMNotConfiguredError(
            f"TRUSTDOC_LLM_BASE_URL must be an http(s) URL such as {DEFAULT_BASE_URL}, got {url!r}"
        )
    return url.strip().rstrip("/")


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8", errors="replace"))
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
    except (ValueError, OSError):
        pass
    return exc.reason if isinstance(exc.reason, str) else f"HTTP {exc.code}"


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
