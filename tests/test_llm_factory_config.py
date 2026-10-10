import socket
import sys

import pytest

from trustdoc.config import AskConfig, Settings
from trustdoc.exceptions import ConfigError, LLMNotConfiguredError
from trustdoc.llm import create_llm, register_backend, supported_providers
from trustdoc.llm.base import GenerationResult, LLMBackend
from trustdoc.llm.factory import BACKENDS
from trustdoc.llm.hf_transformers import TransformersBackend
from trustdoc.llm.ollama import OllamaBackend
from trustdoc.llm.stub import ExtractiveStubLLM


def settings_for(tmp_path, **env):
    values = {"TRUSTDOC_DATA_DIR": str(tmp_path), "TRUSTDOC_LOG_TO_FILE": "false"}
    values.update({f"TRUSTDOC_{k}": str(v) for k, v in env.items()})
    return Settings.from_env(values)


@pytest.fixture
def no_heavy_imports_or_network(monkeypatch):
    """Creating a backend must not import torch/transformers or open a socket."""
    for mod in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, mod, None)

    def boom(*a, **k):
        raise AssertionError("network access during backend construction")

    monkeypatch.setattr(socket.socket, "connect", boom)


# ------------------------------------------------------------ backend selection
def test_factory_selects_the_configured_backend_without_loading_anything(tmp_path, no_heavy_imports_or_network):
    ollama = create_llm(settings_for(tmp_path, LLM_PROVIDER="ollama", LLM_MODEL="qwen2.5:1.5b",
                                     LLM_TEMPERATURE="0.2", LLM_MAX_TOKENS="100", LLM_TIMEOUT_SECONDS="7"))
    assert isinstance(ollama, OllamaBackend) and ollama.model_name == "qwen2.5:1.5b"
    assert ollama.base_url == "http://localhost:11434" and ollama._timeout == 7 and ollama._max_tokens == 100
    hf = create_llm(settings_for(tmp_path, LLM_PROVIDER="Transformers", LLM_MODEL="some/model"))
    assert isinstance(hf, TransformersBackend) and hf._loaded is None
    stub = create_llm(settings_for(tmp_path, LLM_PROVIDER="stub"))
    assert isinstance(stub, ExtractiveStubLLM) and stub.is_stub


def test_custom_base_url_is_honoured(tmp_path):
    b = create_llm(settings_for(tmp_path, LLM_PROVIDER="ollama", LLM_MODEL="m", LLM_BASE_URL="http://gpu-box:11434/"))
    assert b.base_url == "http://gpu-box:11434"


@pytest.mark.parametrize("provider", ["none", "", "NONE"])
def test_no_provider_gives_actionable_error(tmp_path, provider):
    with pytest.raises(LLMNotConfiguredError, match="TRUSTDOC_LLM_PROVIDER") as e:
        create_llm(settings_for(tmp_path, LLM_PROVIDER=provider))
    assert "ollama" in str(e.value)


def test_unknown_provider_lists_the_supported_ones(tmp_path):
    with pytest.raises(LLMNotConfiguredError, match="openai") as e:
        create_llm(settings_for(tmp_path, LLM_PROVIDER="openai"))
    assert "ollama" in str(e.value) and "transformers" in str(e.value)       # never silently falls back to a paid API


def test_backends_without_a_model_are_refused_with_instructions(tmp_path):
    with pytest.raises(LLMNotConfiguredError, match="ollama pull"):
        create_llm(settings_for(tmp_path, LLM_PROVIDER="ollama"))
    with pytest.raises(LLMNotConfiguredError, match="TRUSTDOC_LLM_MODEL"):
        create_llm(settings_for(tmp_path, LLM_PROVIDER="transformers"))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "localhost:11434", "http://", "javascript:alert(1)"])
def test_non_http_base_urls_are_rejected(tmp_path, url):
    with pytest.raises(LLMNotConfiguredError, match="http"):
        create_llm(settings_for(tmp_path, LLM_PROVIDER="ollama", LLM_MODEL="m", LLM_BASE_URL=url))


def test_register_backend_is_an_extension_point(tmp_path):
    class Custom(LLMBackend):
        name = "custom"

        def generate(self, prompt, system_prompt=None, *, max_tokens=None, temperature=None):
            return GenerationResult("x", self.model_name, self.name)

    register_backend("Custom", lambda cfg: Custom(cfg.model or "c"))
    try:
        assert "custom" in supported_providers()
        assert create_llm(settings_for(tmp_path, LLM_PROVIDER="custom")).describe() == "c (custom)"
    finally:
        BACKENDS.pop("custom", None)


def test_importing_the_llm_package_does_not_import_heavy_modules():
    import subprocess
    code = ("import sys; [sys.modules.__setitem__(m, None) for m in ('torch','transformers','numpy')]; "
            "import trustdoc.llm, trustdoc.workflows; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


# ------------------------------------------------------------------- configuration
def test_new_settings_have_cpu_friendly_defaults(tmp_path):
    s = settings_for(tmp_path)
    assert s.llm.provider == "none" and s.llm.model == ""
    assert (s.llm.timeout_seconds, s.llm.max_tokens, s.llm.context_tokens, s.llm.temperature) == (120, 512, 4096, 0.0)
    assert s.ask == AskConfig(5, 6000, 2000)


def test_new_settings_read_from_environment(tmp_path):
    s = settings_for(tmp_path, LLM_MAX_TOKENS=256, LLM_CONTEXT_TOKENS=2048, ASK_MAX_EVIDENCE_ITEMS=3,
                     ASK_MAX_EVIDENCE_CHARS=1500, ASK_MAX_QUESTION_CHARS=300)
    assert (s.llm.max_tokens, s.llm.context_tokens) == (256, 2048)
    assert (s.ask.max_evidence_items, s.ask.max_evidence_chars, s.ask.max_question_chars) == (3, 1500, 300)


@pytest.mark.parametrize("env", [
    {"LLM_MAX_TOKENS": 0}, {"LLM_CONTEXT_TOKENS": 512, "LLM_MAX_TOKENS": 512},
    {"ASK_MAX_EVIDENCE_ITEMS": 0}, {"ASK_MAX_EVIDENCE_CHARS": 100}, {"ASK_MAX_QUESTION_CHARS": 0},
    {"LLM_MAX_TOKENS": "many"},
])
def test_invalid_new_settings_fail_at_startup_without_loading_a_model(tmp_path, no_heavy_imports_or_network, env):
    with pytest.raises(ConfigError):
        settings_for(tmp_path, **env)


def test_provider_names_are_not_validated_at_config_time_so_part1_behaviour_is_unchanged(tmp_path):
    assert settings_for(tmp_path, LLM_PROVIDER="OpenAI").llm.provider == "openai"


def test_settings_dump_contains_no_secret_values(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTDOC_LLM_API_KEY", "sk-should-never-appear")
    assert "sk-should-never-appear" not in str(settings_for(tmp_path, LLM_PROVIDER="ollama", LLM_MODEL="m").to_dict())
