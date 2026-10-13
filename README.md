# PubMed Hybrid Retrieval

A homework assignment from my agentic AI engineering program. It builds a hybrid search system over biomedical research abstracts from the [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) dataset, using [FastEmbed](https://github.com/qdrant/fastembed) for dense and sparse embeddings, [Qdrant](https://qdrant.tech/) as the vector store, and Reciprocal Rank Fusion (RRF) to merge the two rankings.

```
                        ┌─── Dense (BGE-Large) ──────┐
Query ──► Embed Query ──┤                            ├──► RRF Fusion ──► Final Ranked Results
                        └─── Sparse (SPLADE) ────────┘
```

## Why hybrid

- **Dense retrieval** matches on meaning, so "artificial cornea" finds an abstract that never uses those exact words.
- **Sparse retrieval** matches on exact terms, so identifiers like `rs2301228` or `MC3T3` find their abstract when the dense model does not.
- **RRF** combines the two ranked lists using only each document's rank, so the scores never need normalizing. A document found by both retrievers is promoted above one found by only one.

## How it works

1. Load the `pqa_artificial` config of PubMedQA and take the first 100 rows.
2. Build one searchable document per row from the labelled context paragraphs and the conclusion. The question is left out on purpose: it is what users search with, so including it would let every document match its own query.
3. Embed each document twice: `BAAI/bge-large-en-v1.5` (dense, 1024 dimensions) and `prithivida/Splade_PP_en_v1` (sparse).
4. Store both vectors per point in a single Qdrant collection.
5. For a query, run the dense and sparse searches in one batched request.
6. Fuse the two result lists with RRF (`k = 60`) and label each result as `BOTH`, `Dense only`, or `Sparse only`.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run pubmed-hybrid-retrieval
```

This walks through the whole pipeline: it prints dataset statistics, embeds and stores the documents, prints the top dense and sparse results for a set of test queries, and ends with the RRF-fused ranking for "vitamin D".

The first run downloads the dataset and both embedding models from Hugging Face and computes the embeddings, so it takes a while. After that, the embeddings and the Qdrant collection are reused from `.data/` (gitignored). Delete that folder to rebuild from scratch.

## Tests

```sh
uv run pytest
```

- `tests/test_fusion.py` unit-tests the RRF maths and the source labelling.
- `tests/test_search.py` runs the real models against an in-memory Qdrant and checks, for each query in `queries.py`, whether dense and sparse search put the expected document in their top 5. Those expectations record each retriever's blind spots as well as its hits.

## Project layout

```
src/pubmed_hybrid_retrieval/
  __init__.py     main() and the result printers
  data.py         load PubMedQA, explore it, build the document text
  embeddings.py   dense and sparse embedding, cached to disk
  index.py        Qdrant collection setup and point upload
  search.py       dense + sparse query in one batch
  fusion.py       RRF and result labelling
  queries.py      test queries and their expected targets
  paths.py        locations under .data/
tests/
```

## License

[MIT](LICENSE)
