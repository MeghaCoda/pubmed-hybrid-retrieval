This repo is based off of a Google Colab homework assignment I have for my Agentic AI Coding bootcamp. I have written a LangGraph multi-agent RAG once before using another similar assignment. I have 15 years of experience in the Javascript / React ecosystem but I'm fairly new to python. I'm still practicing the basics, so if I get confused make sure to explain syntax and linguistic quirks of python compared to Javascript and Typescript.

These are the initial instructions:

Hybrid Search & Retrieval — Assignment
Problem Statement
Build a hybrid search & retrieval system over biomedical research abstracts using the PubMedQA dataset. Your system must combine two complementary retrieval strategies — dense (semantic) and sparse (lexical) — and fuse their results using Reciprocal Rank Fusion (RRF) so that the final ranking outperforms either method alone.

You will:

Load & clean a subset of PubMedQA
Construct rich, searchable document text from structured fields
Generate dense embeddings and sparse embeddings
Index documents in Qdrant with both vector types
Implement a hybrid search function that queries both indexes in parallel
Implement RRF to merge and re-rank the two result lists
Test and compare all three retrieval modes (dense-only, sparse-only, hybrid)
Architecture
                        ┌─── Dense (BGE-Large) ──────┐
Query ──► Embed Query ──┤                            ├──► RRF Fusion ──► Final Ranked Results         
                        └─── Sparse (SPLADE) ────────┘
Key Concepts
Dense retrieval excels at semantic / conceptual matching (e.g., "physical activity" ↔ "exercise")
Sparse retrieval excels at exact / lexical matching (e.g., drug names like "metformin")
Hybrid search combines both to cover each method's blind spots
RRF merges ranked lists without needing score normalization — a document found by both methods is promoted
💡 Tips
Document construction matters. Think carefully about which fields to include in the document text and which to exclude. The question field is what users will search with — including it in the document would allow trivial self-matching and inflate your results.