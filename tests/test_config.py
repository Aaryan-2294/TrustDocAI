import pytest

from trustdoc.config import Settings
from trustdoc.exceptions import ConfigError


def env(tmp_path, **extra):
    base = {"TRUSTDOC_DATA_DIR": str(tmp_path)}
    base.update({f"TRUSTDOC_{k}": str(v) for k, v in extra.items()})
    return base


def test_defaults(tmp_path):
    s = Settings.from_env(env(tmp_path))
    assert s.storage.db_path == tmp_path.resolve() / "trustdoc.db"
    assert s.storage.uploads_dir == tmp_path.resolve() / "uploads"
    assert s.logging.level == "INFO"
    assert s.retrieval.rrf_k == 60
    assert s.llm.provider == "none"


def test_env_overrides(tmp_path):
    s = Settings.from_env(env(tmp_path, LOG_LEVEL="debug", FINAL_K=3, LLM_PROVIDER="OpenAI",
                              LOG_TO_FILE="no"))
    assert (s.logging.level, s.retrieval.final_k, s.llm.provider) == ("DEBUG", 3, "openai")
    assert s.logging.to_file is False


@pytest.mark.parametrize("bad", [
    {"LOG_LEVEL": "LOUD"},
    {"CHUNK_MAX_CHARS": 100, "CHUNK_OVERLAP_CHARS": 100},
    {"FINAL_K": 50, "CANDIDATE_K": 10},
    {"RRF_K": 0},
    {"LLM_TEMPERATURE": 5},
    {"TOP_K_NOT_INT": 1, "CANDIDATE_K": "many"},
    {"LOG_TO_FILE": "maybe"},
])
def test_invalid_values_raise(tmp_path, bad):
    with pytest.raises(ConfigError):
        Settings.from_env(env(tmp_path, **bad))


def test_dotenv_is_lower_priority_than_env(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text('# comment\nTRUSTDOC_LOG_LEVEL="WARNING"\nTRUSTDOC_FINAL_K=2\n')
    s = Settings.from_env({"TRUSTDOC_DATA_DIR": str(tmp_path), "TRUSTDOC_FINAL_K": "4"},
                          dotenv_path=dotenv)
    assert s.logging.level == "WARNING"
    assert s.retrieval.final_k == 4


def test_ensure_directories_and_to_dict(tmp_path):
    s = Settings.from_env(env(tmp_path / "x"))
    s.ensure_directories()
    assert s.storage.uploads_dir.is_dir() and s.storage.log_dir.is_dir()
    d = s.to_dict()
    assert isinstance(d["storage"]["db_path"], str)
    assert "api_key" not in d["llm"]  # only the env-var NAME is stored, never a key
