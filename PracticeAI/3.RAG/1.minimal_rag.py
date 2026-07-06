"""
Example A — Minimal RAG in Python (all steps)
=============================================

RAG = Retrieval-Augmented Generation. Instead of trusting the LLM's memory, you:
  1. RETRIEVE the most relevant pieces of YOUR data (by meaning, via embeddings)
  2. AUGMENT the prompt by pasting those pieces in as "context"
  3. GENERATE an answer that the model must base ONLY on that context, with
     citations(reference to the source like a link, web page, etc.) — so the answer is grounded and traceable, not made up.

This combines what the earlier topics taught:
  - Topic 1 (KPI Narrator): data from code, language from the LLM.
  - Topic 2 (Embeddings):    find relevant text by meaning, store it in Chroma.

Two phases, clearly separated below:
  - OFFLINE INGEST: embed your documents and store them. Run once (idempotent).
  - RUNTIME QUERY:  for each question, retrieve -> build context -> generate.

The reference example used OpenAI. This uses Gemini's free tier instead (the
concept is identical). Embeddings are computed with Gemini and handed to Chroma
directly, so Chroma never needs its own model.
chroma : vector database

Run:
    ../1.KPI_Narrator/.venv/bin/python 1.minimal_rag.py
"""

import os
import sys
from pathlib import Path

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Reuse the GOOGLE_API_KEY from the sibling KPI Narrator project's .env ---
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

# --- Models + storage location ----------------------------------------------
EMBED_MODEL = "models/gemini-embedding-001"  # turns text into vectors
CHAT_MODEL = "gemini-2.5-flash"              # writes the grounded answer
DB_PATH = str(Path(__file__).parent / "wise_albert_rag")  # Chroma persists here

embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
chat = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

# The knowledge base. In a real system these would be chunks of your docs,
# tickets, wiki pages, etc. Each carries metadata so answers can cite a source.
DOCUMENTS = [
    {
        "id": "c1",
        "text": "Q2 QBR commitment: deliver complete SDK funnel tracking by August 2026.",
        "metadata": {"source_id": "qbr_q2", "type": "commitment"},
    },
    {
        "id": "c2",
        "text": "Week 25: SDK activation fell from 24,000 to 12,000 week-over-week.",
        "metadata": {"source_id": "D-110", "type": "kpi"},
    },
    {
        "id": "c3",
        "text": "Ops note: SDK rollout paused in 2 regions due to pack coverage gap.",
        "metadata": {"source_id": "ops_w25", "type": "ops"},
    },
]


def ingest() -> chromadb.Collection:
    """OFFLINE phase: embed each document and store it in Chroma. Idempotent —
    `upsert` means re-running the script won't create duplicates."""
    chroma = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma.get_or_create_collection("evidence")
    collection.upsert(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS],
        embeddings=embedder.embed_documents([d["text"] for d in DOCUMENTS]),
    )
    return collection


def rag_answer(collection, question: str, top_k: int = 2) -> str:
    """RUNTIME phase: the three RAG steps for one question."""
    # 1. RETRIEVE — find the top_k most relevant chunks by meaning.
    results = collection.query(
        query_embeddings=[embedder.embed_query(question)],
        n_results=top_k,
    )
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    # 2. AUGMENT — paste the retrieved chunks into the prompt as CONTEXT, each
    #    tagged with its source_id so the model can cite where facts came from.
    context = "\n".join(
        f"- {text} (source: {m['source_id']})"
        for text, m in zip(chunks, metas)
    )

    # 3. GENERATE — the system prompt confines the model to the context and
    #    forces "insufficient evidence" when the context doesn't cover it.
    messages = [
        SystemMessage(content=(
            "Answer using ONLY the CONTEXT below. "
            "Cite source_id in parentheses. "
            "If context is insufficient, say 'insufficient evidence'."
        )),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]
    response = chat.invoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)


def main() -> None:
    collection = ingest()  # run the offline step once

    # A question the knowledge base CAN answer (facts live in c1, c2, c3):
    q1 = "Why did SDK activation drop and what did we promise about the funnel?"
    print("Q:", q1)
    print("A:", rag_answer(collection, q1))

    # A question it CANNOT answer — watch the grounding guardrail trigger
    # "insufficient evidence" instead of hallucinating. This is the whole point
    # of RAG: the model answers from retrieved evidence, or admits it can't.
    q2 = "What is the company's 2026 holiday policy?"
    print("\nQ:", q2)
    print("A:", rag_answer(collection, q2))


if __name__ == "__main__":
    main()
