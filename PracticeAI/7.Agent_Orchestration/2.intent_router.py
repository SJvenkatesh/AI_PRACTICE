"""
Example B — Simple intent router
================================

The first job of an orchestrator: given a question, decide WHICH agents to run.
Running every agent every time is slow and expensive. Instead:

    question → classify_intent (LLM) → intents → plan_agents → agent list

The value: if the intent doesn't need `personal_intel`, we never run it —
saving cost and latency. This is the "STEP 1 + STEP 2" of the flow (see
1.orchestration_flow.md).

Provider: free Gemini (flash-lite). One LLM call (the classifier).

Run:
    ../1.KPI_Narrator/.venv/bin/python 2.intent_router.py
"""

import json
import os
import re
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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# Intent → which agents that intent needs. The orchestrator's routing table.
INTENT_TO_AGENTS = {
    "kpi_anomaly":   ["probing", "prediction"],
    "promise_check": ["adoption"],
    "weekly_digest": ["probing", "adoption", "prediction"],
    "precall_brief": ["personal_intel", "adoption"],
    "qbr_prep":      ["adoption", "prediction", "probing"],
}


def _extract_json_object(text: str) -> str:
    """Pull a JSON object out of the model's reply (strip ``` fences if any)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    return body[start:end + 1] if start != -1 and end != -1 else body


def classify_intent(question: str, persona: str) -> list:
    """LLM classifies the question into one or more known intents."""
    resp = llm.invoke([
        SystemMessage(content=(
            "Classify the user question into one or more intents from: "
            "kpi_anomaly, promise_check, weekly_digest, precall_brief, qbr_prep. "
            'Return JSON only: {"intents": ["..."]}'
        )),
        HumanMessage(content=f"Persona: {persona}\nQuestion: {question}"),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return json.loads(_extract_json_object(text)).get("intents", [])


def plan_agents(intents: list) -> list:
    """Map intents → a unique, order-preserving agent list."""
    agents = []
    for intent in intents:
        agents.extend(INTENT_TO_AGENTS.get(intent, []))
    return list(dict.fromkeys(agents))  # dedupe, preserve order


def main() -> None:
    question = "Why did activation drop? Is commitment C-09 at risk?"
    intents = classify_intent(question, persona="cs_lead")
    agents = plan_agents(intents)

    print("Question:", question)
    print("Intents:", intents)
    print("Agents to run:", agents)
    print(
        "\nThe orchestrator will run only these agents. Anything not implied by\n"
        "the intent (e.g. personal_intel) is skipped — saving cost and latency."
    )


if __name__ == "__main__":
    main()
