"""OllamaBackend against a real local HTTP server that mimics Ollama's documented /api/chat contract.

This verifies our HTTP handling, error mapping and timeouts. It is NOT a test against a real
Ollama installation or a real model.
"""

import socket

import pytest

from trustdoc.exceptions import (
    LLMEmptyOutputError,
    LLMNotConfiguredError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from trustdoc.llm.ollama import OllamaBackend


from ask_support import FakeOllama  # noqa: E402


@pytest.fixture
def fake():
    srv = FakeOllama()
    yield srv
    srv.server.shutdown()
    srv.server.server_close()


def backend(fake, **kw):
    return OllamaBackend("m", fake.url, **kw)


def test_successful_chat_request_and_response_mapping(fake):
    b = backend(fake, temperature=0.3, max_tokens=64, context_tokens=2048)
    r = b.generate("the user prompt", "the system prompt")
    assert (r.text, r.model, r.backend, r.finish_reason) == ("Hello [E1]", "m", "ollama", "stop")
    assert (r.prompt_tokens, r.completion_tokens) == (11, 4) and r.elapsed_ms >= 0
    req = fake.requests[0]
    assert req["path"] == "/api/chat" and req["headers"]["Content-Type"] == "application/json"
    assert req["body"] == {
        "model": "m", "stream": False,
        "messages": [{"role": "system", "content": "the system prompt"}, {"role": "user", "content": "the user prompt"}],
        "options": {"temperature": 0.3, "num_predict": 64, "num_ctx": 2048},
    }


def test_per_call_overrides_and_missing_system_prompt(fake):
    backend(fake).generate("p", max_tokens=9, temperature=0.9)
    body = fake.requests[0]["body"]
    assert body["messages"] == [{"role": "user", "content": "p"}]
    assert body["options"]["num_predict"] == 9 and body["options"]["temperature"] == 0.9


def test_length_finish_reason_is_surfaced(fake):
    fake.respond({"model": "m", "message": {"content": "cut off"}, "done_reason": "length"})
    assert backend(fake).generate("p").finish_reason == "length"


def test_missing_model_tells_the_user_how_to_pull_it(fake):
    fake.respond({"error": "model 'm' not found, try pulling it first"}, status=404)
    with pytest.raises(LLMUnavailableError, match="ollama pull m") as e:
        backend(fake).generate("p")
    assert "not found" in str(e.value)


def test_server_error_status_is_a_response_error(fake):
    fake.respond({"error": "kaboom"}, status=500)
    with pytest.raises(LLMResponseError, match="HTTP 500.*kaboom"):
        backend(fake).generate("p")


@pytest.mark.parametrize("payload, exc", [
    ({"error": "model runner crashed"}, LLMResponseError),
    (b"<html>not json</html>", LLMResponseError),
    ([1, 2, 3], LLMResponseError),
    ({"done": True}, LLMResponseError),
    ({"message": {"content": 5}}, LLMResponseError),
    ({"message": {"content": "   \n"}}, LLMEmptyOutputError),
])
def test_malformed_or_empty_responses_are_errors_never_answers(fake, payload, exc):
    fake.respond(payload)
    with pytest.raises(exc):
        backend(fake).generate("p")


def test_slow_server_raises_a_timeout_with_advice(fake):
    fake.respond({"message": {"content": "late"}}, delay=1.5)
    with pytest.raises(LLMTimeoutError, match="TRUSTDOC_LLM_TIMEOUT_SECONDS"):
        backend(fake, timeout_seconds=0.3).generate("p")


def test_unreachable_server_explains_how_to_start_ollama():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                         # a port nobody is listening on
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        OllamaBackend("m", f"http://127.0.0.1:{port}", timeout_seconds=2).generate("p")


def test_environment_proxies_are_ignored_for_the_local_server(fake, monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")     # a dead proxy would break the request if it were used
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert backend(fake).generate("p").text == "Hello [E1]"


def test_constructor_validation():
    with pytest.raises(LLMNotConfiguredError):
        OllamaBackend("  ")
    assert OllamaBackend("m").base_url == "http://localhost:11434"
    assert OllamaBackend("m", "https://example.org:1234/").base_url == "https://example.org:1234"
    assert OllamaBackend(" m ").model_name == "m" and OllamaBackend("m").describe() == "m (ollama)"
