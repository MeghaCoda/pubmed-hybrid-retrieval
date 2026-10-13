import pytest
from qdrant_client import QdrantClient

from pubmed_hybrid_retrieval.fusion import fuse_results
from pubmed_hybrid_retrieval.queries import OUT_OF_CORPUS_QUERY, QUERY_CASES, QueryCase
from pubmed_hybrid_retrieval.search import search

TOP_K = 5

# The cases live in the package so main() can print the same queries; pytest.param
# only adds a readable name to each one in the test output.
CASES = [pytest.param(case, id=case.label) for case in QUERY_CASES]

IN_CORPUS_QUERY = "thioridazine"


def assert_found(result_ids: list, targets: list[int], expected: bool) -> None:
    if expected:
        # <= on sets means "is a subset of": every target must be in the results.
        assert set(targets) <= set(result_ids)
    else:
        # & is set intersection; an empty set is falsy, so this means "no overlap".
        assert not set(targets) & set(result_ids)


# parametrize runs the test once per case, like test.each in Jest.
@pytest.mark.parametrize("case", CASES)
def test_dense_results(client: QdrantClient, case: QueryCase) -> None:
    dense_results, _ = search(client, case.query, top_k=TOP_K)
    assert_found([point.id for point in dense_results], case.targets, case.dense_finds)


@pytest.mark.parametrize("case", CASES)
def test_sparse_results(client: QdrantClient, case: QueryCase) -> None:
    _, sparse_results = search(client, case.query, top_k=TOP_K)
    assert_found(
        [point.id for point in sparse_results], case.targets, case.sparse_finds
    )


def test_search_returns_top_k_ranked_results_with_payload(client: QdrantClient) -> None:
    results = search(client, IN_CORPUS_QUERY, top_k=TOP_K)

    assert len(results) == 2
    for points in results:
        assert len(points) == TOP_K
        scores = [point.score for point in points]
        assert scores == sorted(scores, reverse=True)
        for point in points:
            assert point.payload is not None
            assert set(point.payload) == {"text", "pubid", "question", "final_decision"}


def test_out_of_corpus_query_scores_lower_than_a_real_match(client: QdrantClient) -> None:
    # Search always returns the nearest points, even when nothing is relevant;
    # the only signal that a query has no real match is a low top score.
    dense_miss, sparse_miss = search(client, OUT_OF_CORPUS_QUERY, top_k=1)
    dense_hit, sparse_hit = search(client, IN_CORPUS_QUERY, top_k=1)

    assert dense_miss[0].score < dense_hit[0].score
    assert sparse_miss[0].score < sparse_hit[0].score


def test_fusion_ranks_a_document_found_by_both_retrievers_first(
    client: QdrantClient,
) -> None:
    fused = fuse_results(*search(client, "vitamin D", top_k=10))

    assert fused[0].point_id == 30
    assert fused[0].source == "BOTH"


def test_fusion_keeps_a_document_only_dense_search_finds(client: QdrantClient) -> None:
    # Sparse search misses this one entirely (see the "synonym" case above).
    fused = fuse_results(*search(client, "artificial cornea", top_k=10))

    assert fused[0].point_id == 15
    assert fused[0].source == "Dense only"
