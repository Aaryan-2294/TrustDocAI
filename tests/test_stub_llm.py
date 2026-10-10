from ask_support import make_evidence

from trustdoc.llm.stub import ExtractiveStubLLM
from trustdoc.workflows.citations import analyze_citations
from trustdoc.workflows.prompts import build_prompt


def test_stub_quotes_and_cites_the_top_evidence_and_says_it_is_not_an_llm():
    b = build_prompt("q", [make_evidence(1, "def f():\n    return 1"), make_evidence(2, "other")], max_items=5, max_chars=999)
    llm = ExtractiveStubLLM()
    r = llm.generate(b.user_prompt, b.system_prompt)
    assert llm.is_stub and r.backend == "stub" and "not a language model" in r.text
    assert "[E1]" in r.text and "def f(): return 1" in r.text
    assert analyze_citations(r.text, {"E1", "E2"}).valid_ids == ["E1"]
    assert llm.generate(b.user_prompt, b.system_prompt).text == r.text           # deterministic


def test_stub_without_evidence():
    assert "no evidence" in ExtractiveStubLLM().generate("nothing here").text
