import pytest

from trustdoc.workflows.citations import analyze_citations

IDS = {"E1", "E2", "E3"}


def test_valid_ids_are_recognised_in_first_appearance_order():
    a = analyze_citations("B first [E2], then A [E1].", IDS)
    assert a.valid_ids == ["E2", "E1"] and a.invalid_ids == [] and a.has_valid
    assert a.sanitized_text == "B first [E2], then A [E1]."


def test_grouped_and_adjacent_forms():
    a = analyze_citations("x [E1, E3] y [E2][E1] z [E1; E2]", IDS)
    assert a.valid_ids == ["E1", "E3", "E2"] and a.mentions == {"E1": 3, "E3": 1, "E2": 2}
    assert a.sanitized_text == "x [E1, E3] y [E2][E1] z [E1, E2]"


def test_duplicates_are_counted_once_in_ids_but_all_mentions_recorded():
    a = analyze_citations("a [E1] b [E1] c [E1,E1]", IDS)
    assert a.valid_ids == ["E1"] and a.mentions["E1"] == 4


def test_unknown_ids_are_flagged_and_marked_never_replaced():
    a = analyze_citations("Claim [E9]. Other [E1, E7].", IDS)
    assert a.invalid_ids == ["E9", "E7"] and a.valid_ids == ["E1"]
    assert a.sanitized_text == "Claim [unknown source E9]. Other [E1] [unknown source E7]."
    assert "E9]" not in a.sanitized_text.replace("unknown source E9]", "")


def test_group_with_several_unknown_ids_uses_plural_marker():
    assert analyze_citations("[E8, E9]", IDS).sanitized_text == "[unknown sources E8, E9]"


def test_no_citations_is_reported_accurately():
    a = analyze_citations("The retry mechanism uses exponential backoff.", IDS)
    assert not a.has_valid and a.valid_ids == [] and a.invalid_ids == [] and a.mentions == {}
    assert a.sanitized_text == "The retry mechanism uses exponential backoff."


def test_plain_mentions_and_other_bracket_forms_are_not_citations():
    a = analyze_citations("See E1 and (E2) and [1] and [note E1] and [Evidence1]", IDS)
    assert a.valid_ids == [] and a.invalid_ids == []


def test_case_and_leading_zeros_are_normalised():
    a = analyze_citations("[e1] and [E01]", IDS)
    assert a.valid_ids == ["E1"] and a.sanitized_text == "[E1] and [E1]"


def test_supplied_ids_define_validity_even_for_evidence_the_model_never_saw():
    a = analyze_citations("[E4]", {"E1", "E2"})                          # E4 existed in retrieval but was dropped
    assert a.invalid_ids == ["E4"] and not a.has_valid


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text(text):
    a = analyze_citations(text, IDS)
    assert a.sanitized_text == text and not a.valid_ids
