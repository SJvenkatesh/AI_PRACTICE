"""
Embeddings — Mini Project, Topic 2
==================================

Builds on Topic 1 (1.KPI_Narrator). Topic 1 was *data from code, language from
the LLM*. This adds the other half of most real LLM systems: *embeddings for
retrieval* — turning text into vectors so you can find the most relevant piece
of data by meaning, not keyword.

Two parts:

  A. Basic embeddings — embed a few texts and inspect the vectors (mirrors the
     classic OpenAI `embeddings.create` example, but on Gemini's free tier).
  B. Semantic KPI routing (the enhancement) — given a free-text question, embed
     it, find the most similar KPI by cosine similarity, then hand ONLY that KPI
     to the narrator. Embeddings pick the data; the LLM phrases it. The numbers
     never come from the model.

Note on the reference example: it used OpenAI's `text-embedding-3-small`
(1536 dims). OpenAI's API needs prepaid credit, so this uses Gemini's
`gemini-embedding-001` (3072 dims) instead — the *concept* is identical, only
the provider and vector size differ.

Setup
-----
    # Reuses the venv + .env from ../1.KPI_Narrator (same GOOGLE_API_KEY).
    ../1.KPI_Narrator/.venv/bin/python embeddings_demo.py

    # Or install into any venv:
    pip install langchain-google-genai python-dotenv
    export GOOGLE_API_KEY=...   # free key: https://aistudio.google.com/app/apikey
    python embeddings_demo.py

Docs: https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/
"""

import math
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Reuse the key from the KPI Narrator project (sibling folder) so we don't store
# it in two places — fall back to a local .env if you'd rather keep one here.
try:
    from dotenv import load_dotenv

    for candidate in (
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / "1.KPI_Narrator" / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    pass

# --- Models -----------------------------------------------------------------
EMBED_MODEL = "models/gemini-embedding-001"  # 3072-dim vectors, free tier
CHAT_MODEL = "gemini-2.5-flash"              # same narrator model as Topic 1

# --- The system of record: KPIs live in code, not in the model --------------
# Each KPI has its value/unit/source (the audited data) plus a short natural-
# language description used only for semantic matching.
KPIS = {
    "monthly_recurring_revenue": {
        "value": 482_000, "unit": "USD", "source": "Stripe (May 2026)",
        "describes": "monthly recurring revenue, MRR, subscription sales run rate",
    },
    "active_customers": {
        "value": 1_274, "unit": "accounts", "source": "Postgres prod.customers",
        "describes": "number of active customers, accounts, user base size",
    },
    "net_revenue_retention": {
        "value": 112, "unit": "%", "source": "Finance model v4",
        "describes": "net revenue retention, NRR, expansion and customer loyalty, repeat revenue",
    },
    "support_csat": {
        "value": 4.6, "unit": "out of 5", "source": "Zendesk (last 30d)",
        "describes": "customer support satisfaction, CSAT, help desk quality",
    },
    "monthly_churn_rate": {
        "value": 2.1, "unit": "%", "source": "Finance model v4",
        "describes": "monthly churn rate, customers lost, cancellations, attrition",
    },
}

SYSTEM_STRICT = (
    "You are a financial reporting assistant. "
    "Use ONLY the data provided in the user's message. "
    "Never invent, estimate, or compute numbers that are not explicitly given. "
    'If a requested metric is not present, state "not available".'
)


def _require_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit(
            "GOOGLE_API_KEY is not set.\n"
            "  Get a free key: https://aistudio.google.com/app/apikey\n"
            "  Put it in ../1.KPI_Narrator/.env as  GOOGLE_API_KEY=..."
        )


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors, in pure Python (no numpy).

    1.0 = identical direction (same meaning), 0.0 = unrelated.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def part_a_basic_embeddings(embedder: GoogleGenerativeAIEmbeddings) -> None:
    """Mirror the classic embeddings example: text in, vectors out."""
    section("A. BASIC EMBEDDINGS — text becomes vectors")
    texts = [
        "SDK activation count dropped from 24k to 12k",
        "Weekly install-to-activate funnel declined sharply",
        "The weather in Jakarta is rainy today",
    ]
    # embed_documents takes a list and returns one vector per text.
    vectors = embedder.embed_documents(texts)
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        preview = [round(x, 4) for x in vector[:3]]
        print(f"Text {i}: {len(vector)} dimensions, first 3 values: {preview}")
        print(f"        {text!r}")

    # The first two texts are about the same thing; the third is unrelated.
    # Cosine similarity shows that the vectors "know" this, with no keywords
    # in common ("activation" vs "install-to-activate").
    print("\nCosine similarity (meaning overlap):")
    print(f"  text0 vs text1 (both about activation): {cosine_similarity(vectors[0], vectors[1]):.3f}")
    print(f"  text0 vs text2 (activation vs weather): {cosine_similarity(vectors[0], vectors[2]):.3f}")


def part_b_semantic_kpi_routing(embedder, chat) -> None:
    """The enhancement: route a free-text question to the right KPI by meaning,
    then let the narrator phrase only that KPI's audited number."""
    section("B. SEMANTIC KPI ROUTING — embeddings pick the data, LLM phrases it")

    # Embed each KPI's description ONCE. In a real system you'd cache these.
    keys = list(KPIS.keys())
    kpi_vectors = embedder.embed_documents([KPIS[k]["describes"] for k in keys])

    question = "How are we doing on customer loyalty and repeat revenue?"
    print(f"Question: {question!r}\n")

    # Embed the question and rank KPIs by cosine similarity.
    q_vector = embedder.embed_query(question)
    ranked = sorted(
        ((cosine_similarity(q_vector, kpi_vectors[i]), keys[i]) for i in range(len(keys))),
        reverse=True,
    )
    print("KPI relevance ranking (cosine similarity):")
    for score, key in ranked:
        print(f"  {score:.3f}  {key}")

    best_key = ranked[0][1]
    print(f"\n-> Best match: {best_key}")

    # Hand ONLY the matched KPI's audited data to the narrator. The number comes
    # from the dict; the model only turns it into a sentence.
    kpi = KPIS[best_key]
    data_line = (
        f"{best_key}: {kpi['value']} {kpi['unit']} (source: {kpi['source']})"
    )
    messages = [
        SystemMessage(content=SYSTEM_STRICT),
        HumanMessage(content=(
            f"KPI data:\n{data_line}\n\n"
            f'The user asked: "{question}"\n'
            "Answer in one sentence using only the data above."
        )),
    ]
    answer = chat.invoke(messages).content
    #"  hello world  \n".strip()   # → "hello world"
    answer = answer.strip() if isinstance(answer, str) else str(answer).strip()
    print(f"\nNarrated answer (data from code, words from LLM):\n{answer}")


def main() -> None:
    _require_key()
    embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    chat = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    part_a_basic_embeddings(embedder)
    part_b_semantic_kpi_routing(embedder, chat)

    section("WHY THIS MATTERS")
    print(
        "- Embeddings turn text into vectors whose distance reflects MEANING,\n"
        "  not shared keywords — so 'customer loyalty and repeat revenue'\n"
        "  matched net_revenue_retention without those exact words.\n"
        "- Retrieval (which data) and generation (how to phrase it) are separate\n"
        "  jobs: embeddings select the right audited number, the LLM only writes\n"
        "  the sentence. The model still never sources the figure.\n"
        "- This is the core of RAG: embed your data, embed the question, fetch\n"
        "  the closest pieces, then narrate strictly from what you fetched."
    )


if __name__ == "__main__":
    main()
