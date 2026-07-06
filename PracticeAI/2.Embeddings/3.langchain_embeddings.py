"""
Example D — LangChain embeddings
================================

Same idea as a raw provider SDK, but through LangChain's uniform interface:
  - embed_query(text)      -> one vector
  - embed_documents([...]) -> a list of vectors (batch)

The reference example used OpenAIEmbeddings; swapping providers is a one-line
change — that is the whole point of the LangChain abstraction. Here it's Gemini.

Run:
    ../1.KPI_Narrator/.venv/bin/python 3.langchain_embeddings.py
"""

import os
import sys
from pathlib import Path

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


def main() -> None:
    # To switch providers later, change only this line, e.g.:
    #   from langchain_openai import OpenAIEmbeddings
    #   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector = embeddings.embed_query("Anomaly in churn rate week over week")
    print("single query vector length:", len(vector))

    batch = embeddings.embed_documents([
        "Weekly KPI sheet row D-150 ARPU",
        "Billing feed latency spike on Sunday",
    ])
    print("batch:", len(batch), "vectors, each", len(batch[0]), "dims")


if __name__ == "__main__":
    main()
