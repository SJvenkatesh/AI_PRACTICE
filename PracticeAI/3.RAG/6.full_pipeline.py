"""
Capstone — Full RAG pipeline (all stages combined)
===================================================

Every earlier example was one piece. This wires them into a single pipeline:

    raw docs
      → CHUNK          (Example B: split into retrievable pieces)
      → EMBED + STORE  (Example A/C: vectors in Chroma)
      → HYBRID RETRIEVE(Example D: BM25 keyword + vector meaning -> candidates)
      → RE-RANK        (Example E: cross-encoder keeps the most relevant few)
      → GENERATE       (Example A: grounded answer, cite source_id, else
                        "insufficient evidence")

The script prints each stage so you can SEE the shortlist shrink and sharpen:
  hybrid gives ~several candidates → re-ranker keeps the best 3 → LLM answers.

Provider: free Gemini (embeddings + chat). Re-ranker: free local flashrank.


INGESTION TIME (batch / scheduled)
══════════════════════════════════
Raw doc: 50-page QBR PDF
    → parse text (PyPDF, Unstructured)
    → chunk (500 tokens, 50 overlap)
    → embed each chunk (text-embedding-3-small)
    → upsert to vector DB with metadata

QUERY TIME (per user question)
══════════════════════════════
"Why did activation drop?"
    → embed question (same model!)
    → ANN search in vector index
    → optional: BM25 merge (hybrid)
    → optional: rerank top 20 → top 3
    → assemble prompt:

        SYSTEM: rules + persona
        CONTEXT: chunk1, chunk2, chunk3 (TEXT only)
        QUESTION: user question

    → LLM generates tokens
    → post-process: validate citations, check confidence
    → return to user


Setup (all already installed if you ran the earlier examples):
    ../1.KPI_Narrator/.venv/bin/pip install \
        langchain langchain-community langchain-chroma \
        langchain-google-genai rank_bm25 flashrank python-dotenv
    ../1.KPI_Narrator/.venv/bin/python 6.full_pipeline.py
"""

import os
import sys
from pathlib import Path

from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# EnsembleRetriever moved to langchain_classic in LangChain v1.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # LangChain < 1.0
    from langchain.retrievers import EnsembleRetriever

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

embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
# Build the re-ranker once (loads the small ONNX model), not per query.
reranker = Ranker()

# Raw source documents — multi-paragraph, like real QBR / ops notes. Each has a
# source_id we'll carry all the way to the citation in the final answer.
RAW_DOCS = [
    {"source_id": "qbr_q2", "text": (
        "## SDK Commitments\n\n"
        "We committed to deliver complete SDK funnel tracking by August 2026. "
        "Owner: Product team.\n\n"
        "## Status\n\n"
        "The SDK funnel commitment is currently amber (at risk)."
    )},
    {"source_id": "D-110", "text": (
        "Week 25 KPI report.\n\n"
        "SDK activation fell from 24,000 to 12,000 week-over-week."
    )},
    {"source_id": "ops_w25", "text": (
        "Ops note, week 25.\n\n"
        "SDK rollout was paused in 2 regions due to a pack coverage gap, "
        "which reduced new activations."
    )},
    {"source_id": "D-150", "text": (
        "Billing feed D-150.\n\n"
        "ARPU for the SDK-enabled cohort averaged IDR 44,000."
    )},
    {"source_id": "hr_2026", "text": (
        "HR announcement.\n\n"
        "The 2026 company holiday calendar has been published on the intranet."
    )},
]


def chunk_by_paragraphs(text: str) -> list[str]:
    """Example B: split on blank lines into coherent, retrievable sections."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def build_chunks() -> list[Document]:
    """CHUNK stage: turn each raw doc into paragraph-level Documents, carrying
    the parent's source_id (plus a chunk index) so citations survive."""
    chunks: list[Document] = []
    for doc in RAW_DOCS:
        for i, piece in enumerate(chunk_by_paragraphs(doc["text"])):
            chunks.append(Document(
                page_content=piece,
                metadata={"source_id": doc["source_id"], "chunk": i},
            ))
    return chunks


def build_hybrid_retriever(chunks: list[Document]) -> EnsembleRetriever:
    """EMBED+STORE and wire up HYBRID retrieval (BM25 keyword + vector meaning)."""
    vectorstore = Chroma.from_documents(chunks, embedder)
    vector_r = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    return EnsembleRetriever(retrievers=[bm25, vector_r], weights=[0.4, 0.6])


def rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """RE-RANK stage: score each (query, chunk) pair, keep the best `top_n`."""
    passages = [
        {"id": i, "text": d.page_content, "meta": d.metadata}
        for i, d in enumerate(docs)
    ]
    ranked = reranker.rerank(RerankRequest(query=query, passages=passages))
    # `id` maps back to the position in `docs`.
    return [docs[r["id"]] for r in ranked[:top_n]]


def generate(query: str, context_docs: list[Document]) -> str:
    """GENERATE stage: grounded answer with citations, or 'insufficient evidence'."""
    context = "\n".join(
        f"- {d.page_content} (source: {d.metadata['source_id']})"
        for d in context_docs
    )
    messages = [
        SystemMessage(content=(
            "Answer using ONLY the CONTEXT below. "
            "Cite source_id in parentheses. "
            "If context is insufficient, say 'insufficient evidence'."
        )),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"),
    ]
    resp = chat.invoke(messages)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def answer(query: str, hybrid: EnsembleRetriever, verbose: bool = True) -> str:
    """Run the full pipeline for one question, showing each stage."""
    candidates = hybrid.invoke(query)          # Stage 1: hybrid retrieve
    top_docs = rerank(query, candidates, 3)    # Stage 2: re-rank

    if verbose:
        print(f"\nQ: {query}")
        print(f"  hybrid candidates ({len(candidates)}):")
        for d in candidates:
            print(f"    - [{d.metadata['source_id']}] {d.page_content[:60]}")
        print("  re-ranked top 3:")
        for d in top_docs:
            print(f"    - [{d.metadata['source_id']}] {d.page_content[:60]}")

    return generate(query, top_docs)           # Stage 3: generate


def main() -> None:
    chunks = build_chunks()
    print(f"CHUNK stage: {len(RAW_DOCS)} raw docs -> {len(chunks)} chunks")
    for d in chunks:
        print(f"  [{d.metadata['source_id']}#{d.metadata['chunk']}] {d.page_content[:60]}")

    hybrid = build_hybrid_retriever(chunks)

    # Answerable question — needs facts spread across qbr_q2, D-110, ops_w25.
    q1 = "Why did SDK activation drop and what is the funnel commitment status?"
    print("\n" + "=" * 70)
    print("A:", answer(q1, hybrid))

    # Unanswerable — the knowledge base has no policy detail. Guardrail should fire.
    q2 = "How many vacation days does the 2026 holiday policy grant?"
    print("\n" + "=" * 70)
    print("A:", answer(q2, hybrid))


if __name__ == "__main__":
    main()
