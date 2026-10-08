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
