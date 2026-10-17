"""Controlled vocabularies shared by code, database CHECK constraints and reports.

Every enum is a ``str`` enum so values serialise naturally, and the SQLite
schema is generated from these classes so the two can never drift apart.
"""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    PYTHON = "python"
    MARKDOWN = "markdown"
    PDF = "pdf"


class ChunkType(str, Enum):
    CODE = "code"            # slice of Python source (usually one entity)
    MARKDOWN = "markdown"    # section of a Markdown file
    PDF_TEXT = "pdf_text"    # text span from a PDF page


class EntityType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class ParameterKind(str, Enum):
    """Mirrors ``inspect.Parameter.kind`` names."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"


class ClaimType(str, Enum):
    ENTITY_REFERENCE = "entity_reference"  # "function foo() does ..." -> foo must exist
    PARAMETER = "parameter"                # statements about parameters
    RETURN = "return"                      # statements about return values
    BEHAVIORAL = "behavioral"              # what the code does (Tier 2)
    # added in schema v2 (Part 5)
    SIGNATURE = "signature"                # a documented signature, e.g. `f(a, b=1) -> int`
    CONFIG_DEPENDENCY = "config_dependency"  # env vars / config keys the code is said to use (Tier 2)
    OTHER = "other"                        # supplied claims that fit no other category (Tier 2)


class ClaimOrigin(str, Enum):
    EXISTING_DOC = "existing_doc"
    GENERATED = "generated"


class VerificationTier(str, Enum):
    DETERMINISTIC = "deterministic"  # Tier 1: AST
    SEMANTIC = "semantic"            # Tier 2: evidence-based


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    # added in schema v2: the engine could not check this claim in this run (capability
    # unavailable/disabled, external API, example code). Distinct from "checked but undecided".
    NOT_CHECKABLE = "not_checkable"


class EvidenceStance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    RELATED = "related"


class IssueType(str, Enum):
    # Tier 1
    NONEXISTENT_ENTITY = "nonexistent_entity"
    PARAMETER_NAME_MISMATCH = "parameter_name_mismatch"
    PARAMETER_COUNT_MISMATCH = "parameter_count_mismatch"
    SIGNATURE_DEFAULT_MISMATCH = "signature_default_mismatch"
    RETURN_MISMATCH = "return_mismatch"
    MISSING_DOCUMENTATION = "missing_documentation"
    # Tier 2 - deliberately two distinct types: "contradicted by evidence"
    # is a much stronger statement than "not supported by available evidence".
    CONTRADICTED_CLAIM = "contradicted_claim"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class Severity(str, Enum):
    INFO = "info"          # added in schema v2: unverified / informational, not a proven defect
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER.index(self)


_SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
