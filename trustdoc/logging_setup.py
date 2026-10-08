"""Logging setup. All project loggers live under the ``trustdoc`` namespace."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from trustdoc.config import Settings

ROOT_LOGGER_NAME = "trustdoc"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_HANDLER_TAG = "_trustdoc_handler"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the ``trustdoc`` namespace."""
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure console (+ optional rotating file) logging. Idempotent."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(settings.log_level_int)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.logging.to_file:
        settings.storage.log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                settings.storage.log_dir / settings.logging.file_name,
                maxBytes=settings.logging.max_bytes,
                backupCount=settings.logging.backup_count,
                encoding="utf-8",
            )
        )
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)
    return logger
