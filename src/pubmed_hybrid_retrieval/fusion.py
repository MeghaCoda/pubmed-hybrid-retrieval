from dataclasses import dataclass
from typing import Any

from qdrant_client import models

PointId = models.ExtendedPointId

SOURCE_BOTH = "BOTH"
SOURCE_DENSE = "Dense only"
SOURCE_SPARSE = "Sparse only"


def rank_list(search_result: list[models.ScoredPoint]) -> list[tuple[PointId, int]]:
    """Step 11: turn Qdrant results into (id, rank) pairs, with the best match at rank 1."""
    # enumerate counts from 0 by default; start=1 makes the ranks 1-indexed.
    return [(point.id, rank) for rank, point in enumerate(search_result, start=1)]


def rrf(
    rank_lists: list[list[tuple[PointId, int]]],
    k: int = 60,
    default_rank: int = 1000,
) -> list[tuple[PointId, float]]:
    """Step 11: fuse rank lists into one list of (id, score) pairs, best first."""
    # A dict per list gives a direct id -> rank lookup instead of scanning the pairs.
    rank_lookups = [dict(pairs) for pairs in rank_lists]
    # dict.fromkeys de-duplicates like a Set but keeps first-seen order, so
    # documents with equal scores come out in a predictable order.
    all_ids = dict.fromkeys(point_id for pairs in rank_lists for point_id, _ in pairs)

    scores = {
        point_id: sum(
            # .get(key, fallback) is like lookup[key] ?? fallback in JS.
            1 / (k + lookup.get(point_id, default_rank))
            for lookup in rank_lookups
        )
        for point_id in all_ids
    }
    # scores.items() yields (id, score) pairs; the lambda picks the score to sort by.
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# @dataclass writes the constructor for you from the annotated fields below,
# much like a TypeScript class with public constructor parameters.
@dataclass
class FusedResult:
    point_id: PointId
    score: float
    source: str
    payload: dict[str, Any]


def fuse_results(
    dense_results: list[models.ScoredPoint],
    sparse_results: list[models.ScoredPoint],
) -> list[FusedResult]:
    """Step 12: fuse the two result lists with RRF and label where each result came from."""
    dense_ids = {point.id for point in dense_results}
    sparse_ids = {point.id for point in sparse_results}
    # RRF only returns ids and scores, so keep each point's payload to attach afterwards.
    payloads = {
        point.id: point.payload or {} for point in dense_results + sparse_results
    }

    fused = []
    for point_id, score in rrf([rank_list(dense_results), rank_list(sparse_results)]):
        if point_id in dense_ids and point_id in sparse_ids:
            source = SOURCE_BOTH
        elif point_id in dense_ids:
            source = SOURCE_DENSE
        else:
            source = SOURCE_SPARSE
        fused.append(FusedResult(point_id, score, source, payloads[point_id]))
    return fused
