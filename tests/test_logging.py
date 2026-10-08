import logging

from trustdoc.config import Settings
from trustdoc.logging_setup import get_logger, setup_logging


def test_namespace():
    assert get_logger("db").name == "trustdoc.db"
    assert get_logger("trustdoc.app").name == "trustdoc.app"
    assert get_logger().name == "trustdoc"


def test_setup_is_idempotent_and_writes_file(tmp_path):
    s = Settings.from_env({"TRUSTDOC_DATA_DIR": str(tmp_path), "TRUSTDOC_LOG_LEVEL": "DEBUG"})
    setup_logging(s)
    logger = setup_logging(s)
    ours = [h for h in logger.handlers if getattr(h, "_trustdoc_handler", False)]
    assert len(ours) == 2  # console + file, not duplicated
    assert logger.level == logging.DEBUG
    get_logger("test").info("hello foundation")
    for h in ours:
        h.flush()
    assert "hello foundation" in (s.storage.log_dir / "trustdoc.log").read_text()
    for h in ours:  # release file handle
        logger.removeHandler(h)
        h.close()
