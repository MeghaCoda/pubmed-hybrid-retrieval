from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from pubmed_hybrid_retrieval.data import add_combined_text, load_pubmedqa, load_subset
from pubmed_hybrid_retrieval.embeddings import load_or_create_embeddings
from pubmed_hybrid_retrieval.index import create_collection, upload_points


# A fixture is setup code that pytest hands to any test listing its name as an
# argument. scope="session" builds it once for the whole run (like beforeAll)
# instead of once per test.
@pytest.fixture(scope="session")
def client() -> Iterator[QdrantClient]:
    """A throwaway in-memory Qdrant holding the same points the app indexes."""
    df = add_combined_text(load_subset(load_pubmedqa()))
    df = load_or_create_embeddings(df)
    # ":memory:" keeps the tests away from the on-disk collection in .data/qdrant.
    client = QdrantClient(":memory:")
    create_collection(client)
    upload_points(client, df)
    # Everything before yield is setup; everything after it is teardown.
    yield client
    client.close()
