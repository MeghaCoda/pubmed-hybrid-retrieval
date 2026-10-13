from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

from pubmed_hybrid_retrieval.paths import EMBEDDINGS_CACHE

SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# How many texts go through the model at once. fastembed's default of 256 used
# over 20 GB of memory on our 100 documents; 4 keeps the peak around 2 GB.
BATCH_SIZE = 4


# @cache remembers the return value, so repeated calls reuse one model instead of
# building a new one each time. Wrapping the model in a function (rather than a
# module-level variable) delays loading until the first call, so importing this
# file stays fast.
@cache
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@cache
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


def make_sparse_embedding(texts: list[str]) -> list[SparseEmbedding]:
    """Step 5: embed texts with SPLADE; each result holds token indices and their weights."""
    # .embed() returns a lazy generator, so list() is what actually runs the model.
    return list(get_sparse_model().embed(texts, batch_size=BATCH_SIZE))


def make_dense_embedding(texts: list[str]) -> list[np.ndarray]:
    """Step 5: embed texts with BGE-Large; each result is a 1024-dimensional vector."""
    return list(get_dense_model().embed(texts, batch_size=BATCH_SIZE))


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: add sparse_embedding and dense_embedding columns built from combined_text."""
    texts = df["combined_text"].tolist()
    # Each cell holds a whole object (not a number or string), so the list is wrapped
    # in a Series with dtype=object; pandas' type stubs reject a bare list of objects.
    print(f"\nEmbedding {len(texts)} documents (sparse)...")
    df["sparse_embedding"] = pd.Series(
        make_sparse_embedding(texts), index=df.index, dtype=object
    )
    print(f"Embedding {len(texts)} documents (dense)...")
    df["dense_embedding"] = pd.Series(
        make_dense_embedding(texts), index=df.index, dtype=object
    )
    return df


def load_or_create_embeddings(
    df: pd.DataFrame, cache_path: Path = EMBEDDINGS_CACHE
) -> pd.DataFrame:
    """Step 6: reuse the embeddings saved by an earlier run, or compute and save them."""
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        # Only trust the saved file if it was built from exactly these documents.
        if cached["combined_text"].equals(df["combined_text"]):
            print(f"\nLoaded saved embeddings from {cache_path}")
            return cached
    df = add_embeddings(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Pickle is Python's built-in way to save arbitrary objects to a file.
    df.to_pickle(cache_path)
    return df
