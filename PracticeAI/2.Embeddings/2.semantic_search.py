"""
Example C — Mini semantic search (no vector DB yet)
===================================================

Embed a small set of documents, embed a query, rank documents by cosine
similarity. No database — just vectors and a sort. This is semantic search in
its simplest form: the top hit matches by MEANING, even when the query doesn't
share the document's exact words ("promise" vs "commitment").

Run:
    ../1.KPI_Narrator/.venv/bin/python 2.semantic_search.py
"""

import os
import sys
from pathlib import Path

import numpy as np
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


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed(texts):
    """Return one embedding vector per text."""
    return embedder.embed_documents(texts)


def main() -> None:
    documents = [
        "QBR commitment: deliver SDK funnel tracking by Q3",
        "ARPU increased 17% for SDK-enabled cohort",
        "Ops portal shows rising false positive rate on IM3",
        "Company holiday policy updated for 2026",
    ]
    doc_vectors = embed(documents)

    query = "What did we promise about the SDK funnel?"
    query_vector = embed([query])[0]

    scores = [cosine_similarity(query_vector, dv) for dv in doc_vectors]
    ranked = sorted(zip(scores, documents), reverse=True)

    print(f"Query: {query!r}\n")
    for score, doc in ranked:
        print(f"{score:.3f} | {doc}")
    print(
        "\nTop hit should be the QBR commitment line — even though the query said\n"
        '"promise", not "commitment". Meaning matched, not keywords.'
    )


if __name__ == "__main__":
    main()
