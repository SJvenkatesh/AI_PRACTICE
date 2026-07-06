"""
Example G — Wise Albert-style metadata (production pattern)
===========================================================

Don't embed only raw text — attach METADATA to each chunk. Metadata gives you:
  - filtering: search only within operator=IOH, week=2026-W25, persona=product
  - citations: every hit carries its source_id, so a RAG answer can be traced

This builds on Example E (Chroma). We store several chunks across operators with
rich metadata, then run a similarity search CONSTRAINED by a `where` filter so
only IOH chunks are considered.

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langchain-chroma
    ../1.KPI_Narrator/.venv/bin/python 6.metadata_filtering.py

Persists to ./chroma_wise_albert (gitignored). Re-running is safe (upsert).
"""

import os
import sys
from pathlib import Path

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
DB_PATH = str(Path(__file__).parent / "chroma_wise_albert")

# Each chunk = text + metadata. Chroma metadata values must be str/int/float/bool.
CHUNKS = [
    {
        "id": "chunk_42",
        "text": "Activation fell 50% WoW after SDK pause in 2 regions.",
        "meta": {"source_id": "D-110", "source_type": "kpi_sheet",
                 "operator": "IOH", "week": "2026-W25", "persona": "product"},
    },
    {
        "id": "chunk_43",
        "text": "IOH churn ticked up to 2.4% following the billing migration.",
        "meta": {"source_id": "D-118", "source_type": "kpi_sheet",
                 "operator": "IOH", "week": "2026-W25", "persona": "finance"},
    },
    {
        "id": "chunk_44",
        "text": "TRI activation steady week over week; no anomalies detected.",
        "meta": {"source_id": "D-201", "source_type": "kpi_sheet",
                 "operator": "TRI", "week": "2026-W25", "persona": "product"},
    },
]


def main() -> None:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="kpi_chunks")

    collection.upsert(
        ids=[c["id"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        metadatas=[c["meta"] for c in CHUNKS],
        embeddings=embedder.embed_documents([c["text"] for c in CHUNKS]),
    )

    query = "activation decline root cause"
    q_vector = embedder.embed_query(query)

    print(f"Query: {query!r}\n")

    print("--- WITHOUT filter (all operators) ---")
    _show(collection.query(query_embeddings=[q_vector], n_results=3))

    print("\n--- WITH filter where={'operator': 'IOH'} ---")
    _show(collection.query(
        query_embeddings=[q_vector],
        n_results=3,
        where={"operator": "IOH"},  # only IOH chunks are eligible
    ))
    print(
        "\nThe filtered query never even considers the TRI chunk. In RAG this is\n"
        "what powers scoped retrieval ('answer for IOH only') and citations —\n"
        "every hit carries its source_id so the answer can point back to it."
    )


def _show(results) -> None:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    if not docs:
        print("  (no matches)")
        return
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"  dist {dist:.4f} | {meta['operator']} | cite={meta['source_id']} | {doc}")


if __name__ == "__main__":
    main()
