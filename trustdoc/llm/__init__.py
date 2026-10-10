"""Language-model backends for grounded generation (Part 4).

Everything is lazy: importing this package never loads a model, imports torch,
or touches the network.
"""

from trustdoc.llm.base import GenerationResult, LLMBackend
from trustdoc.llm.factory import create_llm, register_backend, supported_providers

__all__ = ["GenerationResult", "LLMBackend", "create_llm", "register_backend", "supported_providers"]
