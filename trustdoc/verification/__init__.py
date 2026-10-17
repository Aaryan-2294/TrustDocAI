"""Documentation verification, hallucination detection and Trust Scoring (Part 5).

Light imports only; the engine pulls in nothing heavy (retrieval and LLM backends load lazily).
"""

from trustdoc.verification.engine import VerificationEngine
from trustdoc.verification.extraction import extract_docstring_claims, extract_markdown_claims
from trustdoc.verification.render import render_report
from trustdoc.verification.scoring import ScoreInputs, compute_trust_score
from trustdoc.verification.types import ClaimReport, FindingReport, ScoreReport, VerificationReport

__all__ = [
    "VerificationEngine", "VerificationReport", "ClaimReport", "FindingReport", "ScoreReport", "ScoreInputs",
    "compute_trust_score", "extract_markdown_claims", "extract_docstring_claims", "render_report",
]
