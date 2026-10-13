from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models

from pubmed_hybrid_retrieval.paths import QDRANT_PATH

COLLECTION_NAME = "pubmedqa"
DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"
DENSE_VECTOR_SIZE = 1024


def get_client(path: Path = QDRANT_PATH) -> QdrantClient:
    """Step 7: open Qdrant in local mode, storing collections in a folder on disk."""
    # The assignment uses QdrantClient(":memory:"); a path keeps the data between runs.
    return QdrantClient(path=str(path))


def create_collection(client: QdrantClient) -> None:
    """Step 7: create the pubmedqa collection with one dense and one sparse vector per point."""
    # The collection survives between runs, so only create it the first time.
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )


def make_points(df: pd.DataFrame) -> list[models.PointStruct]:
    """Step 8: turn each row into a point holding its payload and both of its vectors."""
    points = []
    # to_dict("records") gives one plain dict per row, like an array of objects in JS.
    # A pandas index can hold strings or dates, not only numbers, so the id comes
    # from enumerate()'s counter instead, which is always an int.
    for position, row in enumerate(df.to_dict("records")):
        sparse = row["sparse_embedding"]
        points.append(
            models.PointStruct(
                id=position,
                payload={
                    "text": row["combined_text"],
                    # pandas stores numbers as numpy types, which are not plain
                    # Python ints; int() converts so the payload can be saved as JSON.
                    "pubid": int(row["pubid"]),
                    "question": row["question"],
                    "final_decision": row["final_decision"],
                },
                vector={
                    # .tolist() turns the numpy arrays into plain Python lists.
                    DENSE_VECTOR_NAME: row["dense_embedding"].tolist(),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
            )
        )
    return points


def upload_points(client: QdrantClient, df: pd.DataFrame) -> None:
    """Step 8: upsert every row into the collection."""
    # Upsert overwrites points whose id already exists, so re-running adds no duplicates.
    client.upsert(collection_name=COLLECTION_NAME, points=make_points(df))
