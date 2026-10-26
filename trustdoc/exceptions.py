"""Project-specific exception hierarchy."""


class TrustDocError(Exception):
    """Base class for all TrustDoc errors."""


class ConfigError(TrustDocError):
    """Invalid or inconsistent configuration."""


class DatabaseError(TrustDocError):
    """Database-level failure (schema mismatch, integrity violation, ...)."""


class DuplicateError(DatabaseError):
    """A uniqueness constraint was violated (e.g. project name already used)."""


class NotFoundError(TrustDocError):
    """A requested record does not exist."""


# ---------------------------------------------------------------- ingestion
class IngestionError(TrustDocError):
    """A source could not be ingested."""


class UnsupportedSourceError(IngestionError):
    """File type is not supported by the MVP (Python, Markdown, text PDF)."""


class InvalidSourceNameError(IngestionError):
    """Source name is empty, absolute, or attempts path traversal."""


class FileTooLargeError(IngestionError):
    """File exceeds the configured ingestion size limit."""


class ParsingError(IngestionError):
    """File content is malformed (e.g. Python syntax error, corrupt PDF)."""


class MissingDependencyError(TrustDocError):
    """An optional dependency required for this operation is not installed."""


# ---------------------------------------------------------------- retrieval
class RetrievalError(TrustDocError):
    """Base class for retrieval / indexing failures."""


class IndexNotBuiltError(RetrievalError):
    """No usable index exists for the project (build it first)."""


class StaleIndexError(RetrievalError):
    """The stored index no longer matches the project's knowledge base."""


class IndexCorruptError(RetrievalError):
    """Index files are missing, unreadable or inconsistent."""


# ---------------------------------------------------------------------- ask / LLM
class InvalidQuestionError(TrustDocError):
    """The question is empty, too long, or contains unusable characters."""


class LLMError(TrustDocError):
    """Base class for language-model backend failures."""


class LLMNotConfiguredError(LLMError):
    """No usable LLM backend/model is configured (set TRUSTDOC_LLM_PROVIDER / TRUSTDOC_LLM_MODEL)."""


class LLMUnavailableError(LLMError):
    """The backend cannot be reached or the model is not installed locally."""


class LLMTimeoutError(LLMError):
    """The backend did not answer within the configured timeout."""


class LLMResponseError(LLMError):
    """The backend answered, but the response was unusable (malformed, error status, too long)."""


class LLMEmptyOutputError(LLMResponseError):
    """The model returned no text."""


# ------------------------------------------------------------------ verification
class VerificationError(TrustDocError):
    """Documentation verification could not be completed."""


class DocumentSelectionError(TrustDocError):
    """A requested document is unknown or is not a Markdown documentation source.

    A usage error, deliberately NOT a VerificationError (which signals that a run itself failed).
    """

# ------------------------------------------------------------------ storage & cleanup
class CleanupError(TrustDocError):
    """Storage or index cleanup failed during deletion."""

