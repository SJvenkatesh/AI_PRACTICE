"""
Example B — Measure similarity (cosine similarity)
==================================================

Cosine similarity = how close two vectors point in the same direction.
  1.0  = identical direction (same meaning)
  0.0  = unrelated
  < 0  = opposite (rare with these embedding models)

The reference example used "pretend" embeddings. Here we get REAL vectors from
Gemini's free embedding model, then score them with the numpy formula.

Run:
    ../1.KPI_Narrator/.venv/bin/python 1.cosine_similarity.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Reuse the GOOGLE_API_KEY from the sibling KPI Narrator project's .env.
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


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    texts = [
        "SDK activation count dropped from 24k to 12k",
        "Weekly install-to-activate funnel declined sharply",
        "The weather in Jakarta is rainy today",
    ]
    # Real embeddings from the API (one vector per text).
    emb_activation_drop, emb_funnel_decline, emb_weather = embedder.embed_documents(texts)

    print("drop vs funnel: ", round(cosine_similarity(emb_activation_drop, emb_funnel_decline), 3))
    print("drop vs weather:", round(cosine_similarity(emb_activation_drop, emb_weather), 3))
    print(
        "\nExpected pattern: 'drop vs funnel' is high (same meaning, different\n"
        "words) and clearly higher than 'drop vs weather'. That gap — not the\n"
        "absolute number — is the core of semantic search."
    )


if __name__ == "__main__":
    main()
