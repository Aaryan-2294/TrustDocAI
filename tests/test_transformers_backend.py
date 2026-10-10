"""TransformersBackend against FAKE `torch`/`transformers` modules.

These tests check our control flow and arguments (lazy loading, chat template, generate kwargs,
error mapping). They do NOT exercise the real libraries or any real model.
"""

import contextlib
import sys
import types

import pytest

from trustdoc.exceptions import (
    LLMEmptyOutputError,
    LLMError,
    LLMNotConfiguredError,
    LLMResponseError,
    LLMUnavailableError,
    MissingDependencyError,
)
from trustdoc.llm.hf_transformers import TransformersBackend


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, i):
        return self.rows[i]


@pytest.fixture
def fake_hf(monkeypatch):
    state = types.SimpleNamespace(prompt_tokens=20, new_tokens=[7, 8, 9], decoded="Generated answer [E1]",
                                  loads=0, generate_calls=[], template_calls=[], template_error=None,
                                  load_error=None, generate_error=None)

    class Tok:
        pad_token_id = None
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            state.template_calls.append(messages)
            if state.template_error:
                raise state.template_error
            return "<templated>"

        def __call__(self, text, return_tensors=None):
            state.last_text = text
            return {"input_ids": FakeTensor([[1] * state.prompt_tokens]), "attention_mask": FakeTensor([[1] * state.prompt_tokens])}

        def decode(self, ids, skip_special_tokens=True):
            state.decoded_ids = list(ids)
            return state.decoded

    class Model:
        def eval(self):
            return self

        def generate(self, **kwargs):
            if state.generate_error:
                raise state.generate_error
            state.generate_calls.append(kwargs)
            return FakeTensor([[1] * state.prompt_tokens + state.new_tokens])

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            state.loads += 1
            if state.load_error:
                raise state.load_error
            return Tok()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name):
            return Model()

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: contextlib.nullcontext()
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer, transformers.AutoModelForCausalLM = AutoTokenizer, AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


def make(**kw):
    return TransformersBackend("org/chat-model", **kw)


def test_generates_with_chat_template_and_greedy_defaults(fake_hf):
    r = make(max_tokens=64, timeout_seconds=30).generate("user prompt", "system prompt")
    assert (r.text, r.model, r.backend, r.finish_reason) == ("Generated answer [E1]", "org/chat-model", "transformers", "stop")
    assert (r.prompt_tokens, r.completion_tokens) == (20, 3) and fake_hf.decoded_ids == [7, 8, 9]
    assert fake_hf.template_calls == [[{"role": "system", "content": "system prompt"}, {"role": "user", "content": "user prompt"}]]
    kw = fake_hf.generate_calls[0]
    assert kw["max_new_tokens"] == 64 and kw["do_sample"] is False and "temperature" not in kw
    assert kw["max_time"] == 30.0 and kw["pad_token_id"] == 2 and "input_ids" in kw


def test_sampling_is_enabled_only_when_temperature_is_positive(fake_hf):
    make(temperature=0.7).generate("p")
    kw = fake_hf.generate_calls[0]
    assert kw["do_sample"] is True and kw["temperature"] == 0.7
    make().generate("p", temperature=0.0, max_tokens=5)
    assert fake_hf.generate_calls[1]["do_sample"] is False and fake_hf.generate_calls[1]["max_new_tokens"] == 5


def test_model_loads_lazily_once(fake_hf):
    b = make()
    assert fake_hf.loads == 0                              # construction loads nothing
    b.generate("a")
    b.generate("b")
    assert fake_hf.loads == 1


def test_missing_libraries_give_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    b = make()                                             # constructing is fine without the libraries
    with pytest.raises(MissingDependencyError, match=r"\.\[retrieval\]"):
        b.generate("p")


def test_falls_back_to_plain_prompt_when_model_has_no_chat_template(fake_hf):
    fake_hf.template_error = ValueError("no chat template")
    make().generate("the question", "sys")
    assert fake_hf.last_text == "System: sys\n\nUser: the question\n\nAssistant:"


def test_prompt_that_cannot_fit_the_context_window_is_rejected_before_generation(fake_hf):
    fake_hf.prompt_tokens = 4000
    with pytest.raises(LLMResponseError, match="TRUSTDOC_ASK_MAX_EVIDENCE_CHARS"):
        make(max_tokens=512, context_tokens=4096).generate("p")
    assert fake_hf.generate_calls == []


def test_finish_reasons(fake_hf):
    fake_hf.new_tokens = [5] * 8
    assert make(max_tokens=8).generate("p").finish_reason == "length"
    fake_hf.new_tokens = [5]
    assert make(max_tokens=8, timeout_seconds=0).generate("p").finish_reason == "time_limit"


def test_empty_decode_is_an_error(fake_hf):
    fake_hf.decoded = "  "
    with pytest.raises(LLMEmptyOutputError):
        make().generate("p")


def test_model_that_cannot_be_loaded_is_reported_as_unavailable(fake_hf):
    fake_hf.load_error = OSError("couldn't connect to huggingface.co")
    with pytest.raises(LLMUnavailableError, match="org/chat-model"):
        make().generate("p")


def test_runtime_failures_during_generation_are_llm_errors(fake_hf):
    fake_hf.generate_error = RuntimeError("out of memory")
    with pytest.raises(LLMError, match="out of memory"):
        make().generate("p")


def test_model_id_is_required():
    with pytest.raises(LLMNotConfiguredError):
        TransformersBackend(" ")
