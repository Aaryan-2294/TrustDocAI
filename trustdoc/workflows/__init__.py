"""Question-answering workflows (Part 4: Ask)."""

from trustdoc.workflows.ask import AskService, create_ask_service
from trustdoc.workflows.citations import CitationAnalysis, analyze_citations
from trustdoc.workflows.prompts import PromptBundle, build_prompt
from trustdoc.workflows.render import render_ask_result
from trustdoc.workflows.types import (
    AskCitation,
    AskEvidence,
    AskResult,
    AskStatus,
    GroundingLevel,
    RetrievalSummary,
)

__all__ = [
    "AskService", "create_ask_service", "AskResult", "AskStatus", "AskCitation", "AskEvidence",
    "GroundingLevel", "RetrievalSummary", "analyze_citations", "CitationAnalysis",
    "build_prompt", "PromptBundle", "render_ask_result",
]
