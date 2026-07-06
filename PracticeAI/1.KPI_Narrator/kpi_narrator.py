"""
KPI Narrator — Mini Project, Topic 1
====================================

Goal: practice the production pattern — *data from code, language from the LLM*.

The numbers live in a plain Python dict (the "system of record"). The LLM is
only allowed to put those numbers into words. It must never invent or compute
new figures. This script demonstrates:

  1. A clean run: full KPI data -> faithful 3-sentence executive summary.
  2. Breaking it on purpose: drop a KPI from the prompt but still ask about it,
     and watch the model hallucinate a plausible-but-fake number.
  3. The fix: a system rule that forces the model to say "not available" for any
     metric it wasn't given.
  4. Stretch: a temperature comparison (0.0 vs 0.9) showing how sampling
     randomness affects wording — and why that's exactly why you don't trust an
     LLM with the actual numbers.

This version uses LangChain (langchain-google-genai), talking to Google's Gemini
API — which has a free tier. LangChain gives us a uniform ChatModel interface
(SystemMessage / HumanMessage in, AIMessage out).

Setup
-----
    pip install langchain langchain-google-genai python-dotenv
    # Get a free key at https://aistudio.google.com/app/apikey
    cp .env.example .env        # then put your key in .env as GOOGLE_API_KEY
    python kpi_narrator.py

Docs: https://python.langchain.com/docs/integrations/chat/google_generative_ai/
"""

import json
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load GOOGLE_API_KEY from a local .env file if python-dotenv is installed.
# NEVER hardcode the key in this file — keep it in .env (which is gitignored).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- Model ------------------------------------------------------------------
# Gemini Flash: fast and on Google's free tier. gemini-2.5-flash-lite is a
# fallback with a higher free-tier request limit if you hit 429s.
MODEL = "gemini-2.5-flash"

# --- The system of record: KPIs live in code, not in the model --------------
# Each metric carries its value, unit, and source so the summary can be audited.
KPIS = {
    "monthly_recurring_revenue": {"value": 482_000, "unit": "USD", "source": "Stripe (May 2026)"},
    "active_customers":          {"value": 1_274,    "unit": "accounts", "source": "Postgres prod.customers"},
    "net_revenue_retention":     {"value": 112,      "unit": "%", "source": "Finance model v4"},
    "support_csat":              {"value": 4.6,      "unit": "out of 5", "source": "Zendesk (last 30d)"},
    "monthly_churn_rate":        {"value": 2.1,      "unit": "%", "source": "Finance model v4"},
}

# --- System prompts ---------------------------------------------------------
# v0: no guardrail at all — just "be helpful". This is what makes a model most
# likely to hallucinate a missing number, because nothing tells it not to.
SYSTEM_NAIVE = (
    "You are a helpful financial reporting assistant. "
    "Always give the user a concrete, confident answer."
)

# v1: a guardrail. Tells the model to use only provided data, but says nothing
# about what to do when a requested metric is absent.
SYSTEM_STRICT = (
    "You are a financial reporting assistant. "
    "Use ONLY the data provided in the user's message. "
    "Never invent, estimate, or compute numbers that are not explicitly given."
)

# v2: the fix. Adds an explicit instruction for missing metrics, which closes
# the hallucination gap demonstrated below.
SYSTEM_WITH_FALLBACK = (
    SYSTEM_STRICT
    + " If a requested metric is not present in the provided data, you MUST "
    'state "not available" for that metric. Do not guess a value.'
)


def _require_key() -> None:
    """Fail early with a clear message if the key is unset."""
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit(
            "GOOGLE_API_KEY is not set.\n"
            "  Get a free key: https://aistudio.google.com/app/apikey\n"
            '  Then put it in .env as  GOOGLE_API_KEY=...'
        )


def _build_prompt(kpis: dict, question: str) -> str:
    """Build the human-message text. Numbers come from the dict."""
    return (
        f"KPI data (JSON):\n{json.dumps(kpis, indent=2)}\n\n"
        f"{question}"
    )


def narrate(model: ChatGoogleGenerativeAI, kpis: dict, question: str, system: str) -> str:
    """Ask the model to answer `question` using only `kpis`.

    Numbers come from the dict we pass in; the model only supplies wording.
    """
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=_build_prompt(kpis, question)),
    ]
    response = model.invoke(messages)
    return _text(response)


def _text(message) -> str:
    """Extract plain text from a LangChain AIMessage.

    `.content` is usually a string, but can be a list of content blocks.
    Handle both.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    _require_key()

    # The narrator model. LangChain reads GOOGLE_API_KEY from the env.
    # temperature=0 keeps the wording stable for the faithful runs.
    narrator = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

    summary_question = "Write a 3-sentence executive summary of these KPIs."

    # 1) Clean run: full data, faithful summary -----------------------------
    section("1. CLEAN RUN — full data, faithful summary")
    print(narrate(narrator, KPIS, summary_question, SYSTEM_STRICT))

    # 2) Break it on purpose ------------------------------------------------
    # Remove net_revenue_retention from the data we send, but demand it anyway
    # under a naive "always give a confident answer" prompt with no guardrail.
    # The HOPE is to provoke a fabricated NRR figure (a hallucination). Whether
    # it happens depends entirely on the model: weaker/older models often invent
    # a plausible number here; a well-aligned model (e.g. Gemini 2.5) may refuse
    # and say it lacks the data. That variability is the whole point — you must
    # not RELY on the model behaving well (step 3 enforces it instead).
    section("2. BROKEN — demand a metric we did NOT provide (no guardrail)")
    kpis_missing = {k: v for k, v in KPIS.items() if k != "net_revenue_retention"}
    broken_question = (
        "The board deck needs the exact net revenue retention percentage for "
        "this period. State the NRR figure as a percentage in one sentence."
    )
    print("(net_revenue_retention was removed from the data sent to the model)")
    print("(watch for an invented % — or a refusal, depending on the model)\n")
    print(narrate(narrator, kpis_missing, broken_question, SYSTEM_NAIVE))

    # 3) The fix ------------------------------------------------------------
    # Same missing-data scenario, but the v2 system prompt requires the model
    # to say "not available" instead of guessing.
    section('3. FIXED — same question, v2 rule forces "not available"')
    print(narrate(narrator, kpis_missing, broken_question, SYSTEM_WITH_FALLBACK))

    # 4) Stretch: temperature comparison ------------------------------------
    # Same prompt, same model, two temperatures. 0.0 is near-deterministic;
    # 0.9 samples more randomly, so the wording varies between runs.
    section("4. STRETCH — same prompt at temperature 0.0 vs 0.9")
    for temp in (0.0, 0.9):
        print(f"--- temperature = {temp} ---")
        sampler = ChatGoogleGenerativeAI(model=MODEL, temperature=temp)
        print(narrate(sampler, KPIS, summary_question, SYSTEM_WITH_FALLBACK))
        print()

    # 5) The takeaway -------------------------------------------------------
    section("WHY THE LLM IS NOT THE SYSTEM OF RECORD")
    print(
        "- The numbers came from a dict in code, with a value, unit, and source\n"
        "  for each metric — that is auditable and reproducible.\n"
        "- When a number was missing (step 2), the model COULD fill the gap with\n"
        "  a plausible fabrication. Some models do; well-aligned ones refuse. You\n"
        "  cannot rely on which — so you enforce the behavior (step 3) instead.\n"
        "- Temperature (step 4) shows the output is sampled, not computed: the\n"
        "  same prompt yields different wording each run. Anything sampled must\n"
        "  not be your source of truth for figures.\n"
        "- Production pattern: keep the numbers in code/DB; let the LLM only\n"
        "  phrase them, and constrain it to say 'not available' when data is\n"
        "  absent."
    )


if __name__ == "__main__":
    main()
