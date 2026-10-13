import pytest
from qdrant_client import models

from pubmed_hybrid_retrieval.fusion import fuse_results, rank_list, rrf


def make_scored_point(point_id: int, score: float) -> models.ScoredPoint:
    return models.ScoredPoint(id=point_id, version=0, score=score)


def test_rank_list_numbers_results_from_one() -> None:
    results = [make_scored_point(7, 0.9), make_scored_point(3, 0.5)]

    assert rank_list(results) == [(7, 1), (3, 2)]


def test_rank_list_of_no_results_is_empty() -> None:
    assert rank_list([]) == []


def test_rrf_sums_reciprocal_ranks_across_lists() -> None:
    fused = rrf([[("a", 1), ("b", 2)], [("b", 1), ("a", 2)]], k=60)

    # Floats rarely compare exactly equal, so approx allows a tiny rounding difference.
    assert dict(fused) == pytest.approx({"a": 1 / 61 + 1 / 62, "b": 1 / 62 + 1 / 61})


def test_rrf_uses_default_rank_for_items_missing_from_a_list() -> None:
    fused = rrf([[("a", 1)], [("b", 1)]], k=60, default_rank=1000)

    assert dict(fused) == pytest.approx(
        {"a": 1 / 61 + 1 / 1060, "b": 1 / 61 + 1 / 1060}
    )


def test_rrf_sorts_best_score_first() -> None:
    fused = rrf([[("a", 1), ("b", 2), ("c", 3)], [("c", 1), ("b", 2), ("a", 3)]])

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_promotes_an_item_found_by_both_lists() -> None:
    # "both" is only second in each list, but beats the item ranked first in one.
    fused = rrf([[("dense_only", 1), ("both", 2)], [("sparse_only", 1), ("both", 2)]])

    assert fused[0][0] == "both"


def test_rrf_returns_each_item_once() -> None:
    fused = rrf([[("a", 1), ("b", 2)], [("a", 1), ("c", 2)]])

    assert sorted(point_id for point_id, _ in fused) == ["a", "b", "c"]


def test_fuse_results_labels_where_each_result_came_from() -> None:
    dense = [make_scored_point(1, 0.9), make_scored_point(2, 0.8)]
    sparse = [make_scored_point(2, 12.0), make_scored_point(3, 9.0)]

    fused = fuse_results(dense, sparse)

    assert {result.point_id: result.source for result in fused} == {
        1: "Dense only",
        2: "BOTH",
        3: "Sparse only",
    }
    assert fused[0].point_id == 2


def test_fuse_results_carries_the_payload_through() -> None:
    dense = [models.ScoredPoint(id=1, version=0, score=0.9, payload={"text": "abc"})]

    fused = fuse_results(dense, [])

    assert fused[0].payload == {"text": "abc"}
