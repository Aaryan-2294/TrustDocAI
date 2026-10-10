"""Create the configured LLM backend (lazy; never loads a model or contacts a server)."""

from __future__ import annotations

from typing import Callable

from trustdoc.config import LLMConfig, Settings
from trustdoc.exceptions import LLMNotConfiguredError
from trustdoc.llm.base import LLMBackend

BackendFactory = Callable[[LLMConfig], LLMBackend]


def _ollama(cfg: LLMConfig) -> LLMBackend:
    from trustdoc.llm.ollama import OllamaBackend

    return OllamaBackend(
        cfg.model, cfg.base_url, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        context_tokens=cfg.context_tokens, timeout_seconds=cfg.timeout_seconds,
    )


def _transformers(cfg: LLMConfig) -> LLMBackend:
    from trustdoc.llm.hf_transformers import TransformersBackend

    return TransformersBackend(
        cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        context_tokens=cfg.context_tokens, timeout_seconds=cfg.timeout_seconds,
    )


def _stub(cfg: LLMConfig) -> LLMBackend:
    from trustdoc.llm.stub import ExtractiveStubLLM

    return ExtractiveStubLLM(cfg.model or "extractive-stub")


BACKENDS: dict[str, BackendFactory] = {
    "ollama": _ollama,
    "transformers": _transformers,
    "stub": _stub,
}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Add or replace a backend (extension point; e.g. a future remote backend)."""
    BACKENDS[name.strip().lower()] = factory


def supported_providers() -> list[str]:
    return sorted(BACKENDS)


def create_llm(settings: Settings) -> LLMBackend:
    """Return the backend selected by ``settings.llm.provider``.

    Raises ``LLMNotConfiguredError`` (with instructions) when no backend is
    selected or the provider is unknown. Constructing a backend is cheap: models
    load and servers are contacted only on the first ``generate`` call.
    """
    provider = (settings.llm.provider or "none").strip().lower()
    if provider in ("", "none"):
        raise LLMNotConfiguredError(
            "No LLM backend is configured. Set TRUSTDOC_LLM_PROVIDER to one of "
            f"{supported_providers()} (e.g. TRUSTDOC_LLM_PROVIDER=ollama TRUSTDOC_LLM_MODEL=qwen2.5:1.5b). "
            "Questions can still be answered with 'insufficient evidence' without a model."
        )
    try:
        factory = BACKENDS[provider]
    except KeyError:
        raise LLMNotConfiguredError(
            f"Unknown LLM provider {provider!r}. Supported providers: {supported_providers()}."
        ) from None
    return factory(settings.llm)
