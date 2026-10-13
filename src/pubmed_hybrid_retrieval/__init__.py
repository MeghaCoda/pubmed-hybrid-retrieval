from typing import Any

from qdrant_client import QdrantClient

from pubmed_hybrid_retrieval.data import (
    add_combined_text,
    explore_dataset,
    explore_subset,
    load_pubmedqa,
    load_subset,
)
from pubmed_hybrid_retrieval.embeddings import load_or_create_embeddings
from pubmed_hybrid_retrieval.fusion import fuse_results
from pubmed_hybrid_retrieval.index import (
    COLLECTION_NAME,
    create_collection,
    get_client,
    upload_points,
)
from pubmed_hybrid_retrieval.queries import OUT_OF_CORPUS_QUERY, QUERY_CASES
from pubmed_hybrid_retrieval.search import search

PREVIEW_LENGTH = 90


def preview(payload: dict[str, Any] | None) -> str:
    """The first characters of a result's text, flattened onto one line."""
    # `payload or {}` falls back to an empty dict when payload is None, like ?? in JS.
    text = (payload or {}).get("text", "")
    # [:n] is a slice, like .slice(0, n).
    return text[:PREVIEW_LENGTH].replace("\n", " ") + "..."


def print_search_results(client: QdrantClient, query: str, top_k: int = 5) -> None:
    """Step 10: print the top dense and sparse results for one query."""
    dense_results, sparse_results = search(client, query, top_k=top_k)
    print(f"\nQuery: {query}")
    for label, results in [("Dense", dense_results), ("Sparse", sparse_results)]:
        print(f"  {label}:")
        for rank, point in enumerate(results, start=1):
            # :>7.3f right-aligns the score in 7 characters with 3 decimals.
            print(
                f"    {rank}. {point.score:>7.3f}  [{point.id}] {preview(point.payload)}"
            )


def print_fused_results(client: QdrantClient, query: str, top_k: int = 10) -> None:
    """Step 12: print the RRF-fused ranking for one query, with each result's source."""
    dense_results, sparse_results = search(client, query, top_k=top_k)
    fused = fuse_results(dense_results, sparse_results)
    print(f"\nRRF fused ranking for: {query}")
    # Fusing two lists can return up to 2 * top_k results, so keep the first top_k.
    for rank, result in enumerate(fused[:top_k], start=1):
        # :<11 pads the label to 11 characters so the columns line up.
        print(
            f"  {rank:>2}. {result.source:<11}  {result.score:.4f}  "
            f"[{result.point_id}] {preview(result.payload)}"
        )


def main() -> None:
    dataset = load_pubmedqa()
    explore_dataset(dataset)

    df = load_subset(dataset)
    explore_subset(df)

    df = add_combined_text(df)
    print(f"\nSample combined_text:\n{df['combined_text'][0]}")

    df = load_or_create_embeddings(df)
    sparse = df["sparse_embedding"][0]
    dense = df["dense_embedding"][0]
    print(f"Sparse embedding: {len(sparse.indices)} non-zero tokens")
    print(f"Dense embedding: {dense.shape[0]} dimensions")

    client = get_client()
    create_collection(client)
    upload_points(client, df)
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection '{COLLECTION_NAME}': {info.points_count} points")
    print(f"  Dense vectors: {info.config.params.vectors}")
    print(f"  Sparse vectors: {info.config.params.sparse_vectors}")

    for query in [case.query for case in QUERY_CASES] + [OUT_OF_CORPUS_QUERY]:
        print_search_results(client, query)

    print_fused_results(client, "vitamin D")
