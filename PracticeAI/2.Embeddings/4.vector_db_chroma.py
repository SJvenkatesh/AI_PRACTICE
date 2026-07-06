"""
Example E — Vector database with Chroma (local, beginner-friendly)
==================================================================

A vector DB stores embeddings and does fast similarity search for you, so you
don't hand-roll cosine over a Python list (Examples B/C). Chroma is local and
file-backed — perfect for learning.

The reference example let Chroma call OpenAI to embed. Here we compute the
vectors ourselves with Gemini and hand them to Chroma (`embeddings=...` on add,
`query_embeddings=...` on query). That keeps the embedding model under our
control and avoids Chroma's default local model. Chroma still does the storage
and nearest-neighbour search.

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langchain-chroma
    ../1.KPI_Narrator/.venv/bin/python 4.vector_db_chroma.py

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


def main() -> None:
    client = chromadb.PersistentClient(path=DB_PATH)
    # No embedding_function: we pass vectors directly, so Chroma never needs to
    # load its own model. It still indexes and searches the vectors for us.
    collection = client.get_or_create_collection(name="qbr_snippets")

    ids = ["doc1", "doc2", "doc3"]
    documents = [
        "We committed to SDK funnel visibility by August",
        "Network cleanliness improved on TRI vs IM3",
        "Regulator briefing scheduled for Komdigi",
    ]
    metadatas = [
        {"source": "qbr_q2", "type": "commitment"},
        {"source": "weekly_kpi", "type": "metric"},
        {"source": "ops_calendar", "type": "event"},
    ]
    # upsert (not add) so re-running the script doesn't error on duplicate ids.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedder.embed_documents(documents),
    )

    query = "What promises did we make on SDK?"
    results = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=3,
    )

    print(f"Query: {query!r}\n")
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]  # Chroma-specific: LOWER distance = closer
    for doc, meta, dist, id in zip(docs, metas, dists, ids):
        print(f"distance {dist:.4f} | {meta['type']:>10} | {doc} | {id}")
    print(
        "\nTop hit is the SDK commitment line. The vector DB handled storage and\n"
        "nearest-neighbour search; note distance here is the inverse of cosine\n"
        "similarity — smaller means more similar."
    )


if __name__ == "__main__":
    main()
