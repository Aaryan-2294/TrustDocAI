"""Model-agnostic text-generation interface.

Backends turn ``(system_prompt, prompt)`` into text. They know nothing about
retrieval, evidence or citations; the Ask workflow owns all of that. Backends
raise the ``LLMError`` family (or ``MissingDependencyError``) and never return
a fabricated answer: an empty model output is an error, not an empty success.

Note: sentence embedders and Cross-Encoders are NOT generative models and must
not be used behind this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    backend: str
    finish_reason: str | None = None      # "stop" | "length" | "time_limit" | None (unknown)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: float = 0.0


class LLMBackend(ABC):
    """A text-generation backend. Implementations must load models lazily."""

    name: str = "base"
    is_stub: bool = False

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a completion. Per-call ``max_tokens``/``temperature`` override defaults."""

    def describe(self) -> str:
        return f"{self.model_name} ({self.name})"
