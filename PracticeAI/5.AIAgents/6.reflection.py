"""
Example F — Reflection (self-critique)
======================================

An agent can review its OWN draft before returning it. A reflection step checks
the draft against the evidence rules and flags any claim that isn't supported by
an observation — catching hallucinations before a human ever sees them.

    draft  → "Activation dropped due to a marketing cut"
    reflect→ "marketing cut is not in the evidence — remove it"
    revise → "Activation dropped 50% (D-110); ops notes cite an SDK pause (ops_w25)"

This aligns with trust goals: catch invented claims before they reach a customer.

Without reflection:

Evidence:
Activation ↓50%

↓

LLM

↓

"It dropped because marketing spent less."

❌ Hallucination

With reflection:

Evidence:
Activation ↓50%

↓

Draft

↓

Critique

↓

Unsupported sentence detected

↓

Remove it

↓

Final Answer

Reflection is valuable in tasks where factual accuracy matters, such as:

RAG applications: Ensure every statement is grounded in retrieved documents.
SQL agents: Verify summaries match query results.
Customer support: Avoid promising features or policies not found in the knowledge base.
Financial reports: Check calculations and ensure numbers match the source data.
Medical or legal assistants: Flag unsupported assertions before presenting them.


Run:
    ../1.KPI_Narrator/.venv/bin/python 6.reflection.py
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# The evidence the agent actually gathered from tools (the ground truth).
EVIDENCE = [
    {"fact": "SDK activations fell from 24,000 to 12,000 (50% WoW)", "source_id": "D-110"},
    {"fact": "SDK rollout paused in 2 regions", "source_id": "ops_w25"},
]


def reflect_and_revise(draft: str) -> str:
    """Critique the draft against the evidence and return a corrected version."""
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a strict reviewer. Check the DRAFT against the EVIDENCE. "
            "Remove or correct any claim not supported by the evidence. Every "
            "number and cause must trace to an evidence item, cited by source_id. "
            "Return ONLY the revised answer."
        )),
        HumanMessage(content=(
            f"EVIDENCE:\n{json.dumps(EVIDENCE, indent=2)}\n\n"
            f"DRAFT:\n{draft}"
        )),
    ])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def main() -> None:
    # A draft with a fabricated cause ("marketing cut") that the evidence does
    # NOT support — this is exactly what reflection should catch.
    draft = (
        "SDK activation dropped by 50% last week, primarily because the "
        "marketing budget was cut. We expect it to recover next quarter."
    )
    print("DRAFT (contains an unsupported claim):")
    print(" ", draft)

    revised = reflect_and_revise(draft)
    print("\nREVISED (after self-critique against evidence):")
    print(" ", revised)
    print(
        "\nThe 'marketing cut' claim and the unfounded 'recover next quarter'\n"
        "forecast are gone — neither is in the evidence. What remains is grounded\n"
        "and cited. Reflection is a cheap second pass that raises trust."
    )


if __name__ == "__main__":
    main()
