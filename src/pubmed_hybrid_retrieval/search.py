from qdrant_client import QdrantClient, models

from pubmed_hybrid_retrieval.embeddings import get_dense_model, get_sparse_model
from pubmed_hybrid_retrieval.index import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


def search(
    client: QdrantClient, query_text: str, top_k: int = 10
) -> list[list[models.ScoredPoint]]:
    """Step 9: run the query against both indexes; returns [dense_results, sparse_results]."""
    # query_embed() returns a lazy generator even for one text, so list(...)[0]
    # runs the model and takes the single result.
    dense = list(get_dense_model().query_embed(query_text))[0]
    sparse = list(get_sparse_model().query_embed(query_text))[0]

    # One round trip carrying both searches; responses come back in request order.
    responses = client.query_batch_points(
        collection_name=COLLECTION_NAME,
        requests=[
            models.QueryRequest(
                query=dense.tolist(),
                using=DENSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
            ),
            models.QueryRequest(
                query=models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
                using=SPARSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
            ),
        ],
    )
    return [response.points for response in responses]
