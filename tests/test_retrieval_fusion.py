import math

import pytest

from trustdoc.retrieval import RankedHit, RankedList, reciprocal_rank_fusion


def rl(name, ids, scores=None):
    scores = scores or [1.0] * len(ids)
    return RankedList(name, [RankedHit(cid, rank, s) for rank, (cid, s) in enumerate(zip(ids, scores), 1)])


def by_id(fused):
    return {f.chunk_id: f for f in fused}


def test_single_list_keeps_order_and_scores_are_one_over_k_plus_rank():
    fused = reciprocal_rank_fusion([rl("a", [7, 3, 9])], k=60)
    assert [f.chunk_id for f in fused] == [7, 3, 9] and [f.rank for f in fused] == [1, 2, 3]
    assert [f.rrf_score for f in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_two_lists_fuse_with_hand_computed_scores():
    a, b = rl("bm25", [1, 2, 3]), rl("faiss", [3, 1, 4])
    f = by_id(reciprocal_rank_fusion([a, b], k=60))
    assert f[1].rrf_score == pytest.approx(1 / 61 + 1 / 62)   # rank 1 in A, rank 2 in B
    assert f[3].rrf_score == pytest.approx(1 / 63 + 1 / 61)   # rank 3 in A, rank 1 in B
    assert f[2].rrf_score == pytest.approx(1 / 62)            # only A, rank 2
    assert f[4].rrf_score == pytest.approx(1 / 63)            # only B, rank 3
    assert [x.chunk_id for x in reciprocal_rank_fusion([a, b], k=60)] == [1, 3, 2, 4]


def test_chunk_in_both_lists_beats_chunk_that_tops_only_one():
    # chunk 5 is rank 1 in A only; chunk 6 is rank 4 in both lists
    a = rl("a", [5, 1, 2, 6])
    b = rl("b", [7, 8, 9, 6])
    ranked = [f.chunk_id for f in reciprocal_rank_fusion([a, b], k=60)]
    assert ranked[0] == 6  # 2/64 > 1/61: agreement of two lists outranks a single top hit
    f = by_id(reciprocal_rank_fusion([a, b], k=60))
    assert f[6].rrf_score == pytest.approx(2 / 64) and f[6].rrf_score > f[5].rrf_score


def test_contributions_record_rank_native_score_and_share_per_list():
    f = by_id(reciprocal_rank_fusion([rl("bm25", [1], [12.5]), rl("faiss", [9, 1], [0.9, 0.4])], k=10))[1]
    got = {c.list_name: (c.rank, c.score, c.rrf_contribution) for c in f.contributions}
    assert got == {"bm25": (1, 12.5, pytest.approx(1 / 11)), "faiss": (2, 0.4, pytest.approx(1 / 12))}
    assert f.rrf_score == pytest.approx(1 / 11 + 1 / 12)


def test_native_score_magnitudes_never_influence_fusion():
    low = reciprocal_rank_fusion([rl("a", [1, 2], [0.001, 0.0009]), rl("b", [2, 1], [999.0, 1.0])])
    high = reciprocal_rank_fusion([rl("a", [1, 2], [50.0, 1.0]), rl("b", [2, 1], [0.2, 0.1])])
    assert [(f.chunk_id, f.rrf_score) for f in low] == [(f.chunk_id, f.rrf_score) for f in high]


def test_k_changes_the_score():
    assert reciprocal_rank_fusion([rl("a", [1])], k=1)[0].rrf_score == pytest.approx(1 / 2)
    assert reciprocal_rank_fusion([rl("a", [1])], k=1000)[0].rrf_score == pytest.approx(1 / 1001)


def test_ties_are_deterministic_and_independent_of_list_order():
    a, b = rl("a", [2, 1]), rl("b", [1, 2])        # chunks 1 and 2 get identical scores
    order1 = [f.chunk_id for f in reciprocal_rank_fusion([a, b])]
    order2 = [f.chunk_id for f in reciprocal_rank_fusion([b, a])]
    assert order1 == order2 == [1, 2]              # tie -> lower chunk id
    assert order1 == [f.chunk_id for f in reciprocal_rank_fusion([a, b])]  # repeatable


def test_tie_on_score_prefers_better_best_rank_then_id():
    # 10/30 tie at rank 1 (1/61); 20/40 tie at rank 2 (1/62)
    fused = reciprocal_rank_fusion([rl("a", [30, 20]), rl("b", [10, 40])])
    assert [f.chunk_id for f in fused] == [10, 30, 20, 40]


def test_duplicate_chunk_within_one_list_counts_best_rank_once():
    dup = RankedList("a", [RankedHit(1, 3, 0.1), RankedHit(1, 1, 0.9), RankedHit(2, 2, 0.5)])
    f = by_id(reciprocal_rank_fusion([dup], k=60))
    assert f[1].rrf_score == pytest.approx(1 / 61) and len(f[1].contributions) == 1


def test_empty_inputs_and_validation():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([rl("a", []), rl("b", [])]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([rl("a", [1])], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([rl("a", [1]), rl("a", [2])])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([RankedList("a", [RankedHit(1, 0, 1.0)])])


def test_fusion_scores_sum_exactly():
    fused = reciprocal_rank_fusion([rl("a", [1]), rl("b", [1]), rl("c", [1])], k=60)
    assert fused[0].rrf_score == math.fsum([1 / 61] * 3)
