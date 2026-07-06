"""
Example B — Simple multi-agent (sequential handoff)
===================================================

Multi-agent = several LLMs, each with its OWN role, system prompt, and output
shape, cooperating on a task. The simplest topology is a sequential handoff:

    Agent 1 (Probing)  → signals
        → Agent 2 (Adoption)  → promise risk
            → Agent 3 (Narrative) → human-readable summary

Each agent's output becomes the next agent's input. This is like a pipeline of
specialists — one finds anomalies, one checks commitments, one writes the story.

Why split into agents instead of one big prompt? Each role stays small and
focused (easier to tune/debug), and you can swap or reuse a single agent without
touching the others.

Provider: free Gemini (flash-lite — this makes several calls per run).

User
   │
   ▼
Agent 1
   │
   ▼
Agent 2
   │
   ▼
Agent 3
   │
   ▼
Answer


Run:
    ../1.KPI_Narrator/.venv/bin/python 1.sequential_handoff.py
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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)


def _text(content) -> str:
    """Join Gemini content blocks (thinking + text) into a plain string."""
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def call_llm(system: str, user: str) -> str:
    """One LLM turn with a role (system) and an input (user)."""
    return _text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)


# ── AGENT 1: Probing — find signals in the KPI data ─────────────────────────
probing_system = (
    "You are the Probing Agent. Analyze KPI data and flag anomalies. "
    'Return JSON only: {"signals": [...], "severity": "green|amber|red"}'
)

# ── AGENT 2: Adoption — assess commitment risk from the signals ─────────────
adoption_system = (
    "You are the Adoption Agent. Given probing signals + commitment data, "
    'assess promise risk. Return JSON: {"at_risk_commitments": [...], "summary": "..."}'
)

# ── AGENT 3: Narrative — write for a human (CS Lead) ────────────────────────
narrative_system = (
    "You are the Narrative Agent for the CS Lead. Use ONLY the evidence below. "
    "Cite metric/commitment IDs. Max 4 sentences."
)


def main() -> None:
    kpi_data = {"D-110": {"value": 12000, "prior": 24000}}
    commitments = {"C-09": {"text": "SDK funnel by Aug", "status": "amber"}}

    # Handoff 1 -> 2 -> 3. Each print shows the shape changing per role.
    probing_out = call_llm(probing_system, f"KPI DATA:\n{json.dumps(kpi_data)}")
    print("── Probing Agent ──\n", probing_out, "\n")

    adoption_out = call_llm(
        adoption_system,
        f"PROBING OUTPUT:\n{probing_out}\n\nCOMMITMENTS:\n{json.dumps(commitments)}",
    )
    print("── Adoption Agent ──\n", adoption_out, "\n")

    final = call_llm(
        narrative_system,
        f"EVIDENCE:\nProbing: {probing_out}\nAdoption: {adoption_out}",
    )
    print("── Narrative Agent (final) ──\n", final)
    print(
        "\nThree specialists in a chain: Probing found the anomaly, Adoption tied\n"
        "it to a commitment, Narrative wrote it for a human. Output of one fed the\n"
        "next — a sequential handoff."
    )


if __name__ == "__main__":
    main()


"""
Output:

(wise_albert_env) venkatesh@venkatesh:~/WiseAlbert/PracticeAI/6.Multi_Agent_Systems$ ../1.KPI_Narrator/.venv/bin/python 1.sequential_handoff.py 
── Probing Agent ──
 ```json
{
  "signals": [
    {
      "metric": "D-110",
      "value": 12000,
      "prior": 24000,
      "description": "Value for D-110 is 50% lower than the prior period.",
      "anomaly_type": "significant_decrease"
    }
  ],
  "severity": "red"
}
``` 

── Adoption Agent ──
 ```json
{
  "at_risk_commitments": [
    {
      "commitment_id": "C-09",
      "text": "SDK funnel by Aug",
      "reasoning": "The significant decrease in metric D-110 (50% lower than prior period) could indicate a problem with the SDK or its integration, which is directly relevant to the 'SDK funnel' commitment. The 'amber' status suggests this commitment is already facing some challenges, and the new signal increases the risk of it not being met."
    }
  ],
  "summary": "The 'SDK funnel by Aug' commitment is at high risk due to a significant 50% decrease in metric D-110. This metric likely relates to SDK performance or usage, directly impacting the commitment's success. The existing 'amber' status further elevates the concern."
}
``` 

── Narrative Agent (final) ──
 Metric D-110 has seen a significant 50% decrease compared to the prior period, triggering a "red" severity alert. This drop in D-110 is directly impacting the "SDK funnel by Aug" commitment (C-09), placing it at high risk. The decrease suggests potential issues with the SDK or its integration,which are critical for the commitment's success. The existing "amber" status for C-09 further amplifies the concern.

Three specialists in a chain: Probing found the anomaly, Adoption tied
it to a commitment, Narrative wrote it for a human. Output of one fed the
next — a sequential handoff.

"""