"""
Example D — Hybrid search (keyword + semantic)
==============================================

Problem: pure vector (semantic) search can MISS exact tokens like "D-110" or
"IM3" — embeddings capture meaning, not literal IDs. Pure keyword search misses
paraphrases ("activation declined" vs "activation fell").

Hybrid search runs BOTH and merges the results:
    query "D-110 activation IM3"
      ├── BM25 (keyword) ─→ nails exact "D-110", "IM3"
      └── Vector (meaning) ─→ finds "activation declined / fell"
      └─→ merge & re-weight → best of both


                User Query
                     │
        "D-110 activation IM3"
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
  BM25 Retriever          Vector Retriever
 (exact keywords)      (semantic similarity)
         │                       │
         ▼                       ▼
     Documents A             Documents B
         │                       │
         └───────────┬───────────┘
                     ▼
              Merge & Rank
                     ▼
              Final Documents


LangChain's EnsembleRetriever combines retrievers with tunable weights.

Provider swapped from OpenAI to free Gemini for the vector half. The BM25 half
is pure Python (no API).

Example

Assume your database contains these documents:

Doc1:
Device D-110 failed during IM3 testing.

Doc2:
Activation declined by 20%.

Doc3:
Activation fell after deployment.

Doc4:
Device X-200 passed IM3 testing.

User query:

D-110 activation IM3
BM25 returns

Because it matches exact words:

1. Doc1  ← contains D-110 and IM3
2. Doc4  ← contains IM3
3. Doc2
Vector search returns

Because it understands meaning:

1. Doc3  ← "activation fell"
2. Doc2  ← "activation declined"
3. Doc1
Hybrid merge

The combined ranking might become:

1. Doc1  ← exact IDs + relevant
2. Doc3  ← semantically relevant
3. Doc2  ← semantically relevant
4. Doc4

This gives you the benefits of both approaches.

How LangChain does this

LangChain provides EnsembleRetriever.

from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

When you call:

docs = ensemble.invoke("D-110 activation IM3")

it:

Sends the query to the BM25 retriever.
Sends the same query to the vector retriever.
Collects both result sets.
Combines and re-ranks them using the specified weights.
Returns a single ranked list of documents.
What do the weights mean?
weights=[0.7, 0.3]

means:

70% importance to keyword search.
30% importance to semantic search.

If your documents contain many product codes, ticket numbers, or IDs, you might favor BM25:

weights=[0.8, 0.2]

If your users mostly ask natural-language questions with paraphrases, you might favor semantic search:

weights=[0.3, 0.7]

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langchain langchain-community rank_bm25
    ../1.KPI_Narrator/.venv/bin/python 4.hybrid_search.py
"""

import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# EnsembleRetriever moved to langchain_classic in LangChain v1; fall back to the
# old location for older installs.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # LangChain < 1.0
    from langchain.retrievers import EnsembleRetriever

try:
    from dotenv import load_dotenv

    for _p in (Path(__file__).parent / ".env",
               Path(__file__).parent.parent / "1.KPI_Narrator" / ".env"):
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass

if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY not set. Put it in ../1.KPI_Narrator/.env")


def main() -> None:
    # Corpus with exact IDs (D-110, IM3) AND paraphrasable meaning.
    texts = [
        "KPI D-110: activations fell to 12k from 24k prior week.",
        "IM3 network cleanliness improved after the pack refresh.",
        "Commitment C-09: SDK funnel tracking — status amber.",
        "Weekly install-to-activate funnel declined sharply.",
        "Device X-200 failed during IM3 testing.",
    ]
    metadatas = [
        {"source_id": "D-110"},
        {"source_id": "IM3-note"},
        {"source_id": "C-09"},
        {"source_id": "funnel-w25"},
        {"source_id": "D-110-test"},
    ]

    # Keyword half — exact token matching (BM25). Pure Python via rank_bm25.
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = 2

    # Semantic half — meaning via Gemini embeddings in Chroma.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Merge. weights sum need not be 1; they scale each retriever's contribution.
    # Tune these on your own eval set.
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6],
    )

    query = "D-110: activation drop"
    print("Q:", query, "\n")

    print("--- BM25 only (keyword) ---")
    for d in bm25.invoke(query):
        print(f"  {d.metadata['source_id']:>10} | {d.page_content}")

    print("\n--- Vector only (meaning) ---")
    for d in vector_retriever.invoke(query):
        print(f"  {d.metadata['source_id']:>10} | {d.page_content}")

    print("\n--- Hybrid (merged & re-weighted) ---")
    for d in hybrid.invoke(query):
        print(f"  {d.metadata['source_id']:>10} | {d.page_content}")

    print(
        "\nBM25 locks onto the exact 'D-110'; vector search also surfaces the\n"
        "paraphrased 'funnel declined'. Hybrid gives you both — important for\n"
        "metric IDs (D-150), operator names (IM3, TRI), AND 'why' questions."
    )


if __name__ == "__main__":
    main()
