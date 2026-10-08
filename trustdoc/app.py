"""Application bootstrap: the single place that wires settings, logging and DB."""

from __future__ import annotations

from dataclasses import dataclass

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.logging_setup import get_logger, setup_logging


@dataclass
class AppContext:
    settings: Settings
    db: Database


def bootstrap(settings: Settings | None = None) -> AppContext:
    """Create directories, configure logging and make sure the schema exists."""
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    setup_logging(settings)
    db = Database(settings.storage.db_path)
    version = db.initialize()
    get_logger("app").info("TrustDoc ready (db=%s, schema v%d)", settings.storage.db_path, version)
    return AppContext(settings=settings, db=db)
