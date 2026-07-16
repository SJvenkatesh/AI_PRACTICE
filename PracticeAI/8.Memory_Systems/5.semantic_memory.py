"""
Example E — Semantic memory (vector DB as memory)
=================================================

The memory types so far store text verbatim (full history, window, summary,
user facts). Semantic memory stores past exchanges as EMBEDDINGS and retrieves
them by MEANING — so a new question pulls back relevant old memories even if the
wording differs. This is literally RAG (Topic 3) pointed at your own
conversation history instead of documents.

    end of session  → embed + store the important exchange (with metadata)
    next session    → embed the new question → retrieve the closest memories
                     → inject them into the prompt as "RECALLED MEMORIES"

Metadata (user_id, type, date, commitment_id) lets you scope recall to one user
and cite when/where a memory came from.

Provider: free Gemini embeddings. No chat call (we show the recall step).

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langchain-chroma   # already installed earlier
    ../1.KPI_Narrator/.venv/bin/python 5.semantic_memory.py

Persists to ./chroma_memory (gitignored). Re-running is safe (upsert).
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
DB_PATH = str(Path(__file__).parent / "chroma_memory")

# Episodic memories saved at the end of past sessions.
MEMORIES = [
    {"id": "mem_s1_t4",
     "text": "User flagged C-09 SDK funnel commitment as at risk during IOH prep.",
     "meta": {"user_id": "cs_lead_1", "type": "episodic", "date": "2026-07-05", "commitment_id": "C-09"}},
    {"id": "mem_s1_t7",
     "text": "User escalated the D-110 activation drop to the Product team.",
     "meta": {"user_id": "cs_lead_1", "type": "episodic", "date": "2026-07-05", "commitment_id": "D-110"}},
    {"id": "mem_s2_t2",
     "text": "Different user asked about the 2026 holiday calendar.",
     "meta": {"user_id": "other_user", "type": "episodic", "date": "2026-07-06"}},
]


def main() -> None:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("agent_memory") # Collection name is agent_memory

    # Save memories (embeddings computed by Gemini, stored in Chroma).
    collection.upsert(
        ids=[m["id"] for m in MEMORIES],
        documents=[m["text"] for m in MEMORIES],
        metadatas=[m["meta"] for m in MEMORIES],
        embeddings=embedder.embed_documents([m["text"] for m in MEMORIES]),
    )

    # Next session: recall by meaning, scoped to this user.
    query = "What did we discuss about SDK funnel commitments?"
    results = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=2,
        where={"user_id": "cs_lead_1"},   # only THIS user's memories
    )

    print(f"Query: {query!r}\n")
    print("RECALLED MEMORIES (would be injected into the system prompt):")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        cite = meta.get("commitment_id", meta.get("date"))
        print(f"  - {doc}  ({cite})")
    print(
        "\nThe SDK-funnel memory was recalled by MEANING (the query didn't say\n"
        "'C-09'), and the other user's holiday memory was excluded by the\n"
        "user_id filter. Same tech as RAG — used as long-term episodic memory."
    )


if __name__ == "__main__":
    main()

"""
output:

venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 5.semantic_memory.py
Query: 'What did we discuss about SDK funnel commitments?'

RECALLED MEMORIES (would be injected into the system prompt):
  - User flagged C-09 SDK funnel commitment as at risk during IOH prep.  (C-09)
  - User escalated the D-110 activation drop to the Product team.  (D-110)

The SDK-funnel memory was recalled by MEANING (the query didn't say
'C-09'), and the other user's holiday memory was excluded by the
user_id filter. Same tech as RAG — used as long-term episodic memory.

"""