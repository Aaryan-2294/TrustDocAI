import json

from trustdoc.app import bootstrap
from trustdoc.cli import main
from trustdoc.logging_setup import ROOT_LOGGER_NAME
import logging


def _cleanup_logging():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_bootstrap_creates_dirs_and_db(tmp_path):
    from trustdoc.config import Settings
    s = Settings.from_env({"TRUSTDOC_DATA_DIR": str(tmp_path / "d")})
    ctx = bootstrap(s)
    try:
        assert s.storage.db_path.is_file() and s.storage.uploads_dir.is_dir()
        assert ctx.db.schema_version() >= 1
    finally:
        _cleanup_logging()


def test_cli_init_and_info(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRUSTDOC_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.chdir(tmp_path)
    try:
        assert main(["init"]) == 0
        capsys.readouterr()
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out[out.index("{"):])
        assert payload["schema_version"] == 1
        assert payload["table_counts"]["projects"] == 0
    finally:
        _cleanup_logging()


def test_cli_reports_config_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRUSTDOC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRUSTDOC_LOG_LEVEL", "LOUD")
    assert main(["info"]) == 2
    assert "LOG_LEVEL" in capsys.readouterr().out
