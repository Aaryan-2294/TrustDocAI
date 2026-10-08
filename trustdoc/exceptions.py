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
