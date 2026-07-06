"""
Example E — Planning (multi-step upfront)
=========================================

ReAct (Examples B-D) decides the next step one observation at a time — great for
exploratory questions. For a complex, well-understood task you often do better to
PLAN the whole sequence first, then execute it. This is "plan-and-execute":

    QUESTION
      → LLM writes a PLAN (ordered list of tool calls)
      → EXECUTE each step, collecting observations
      → LLM SYNTHESISES a final answer from all observations

When to plan: complex tasks with a knowable path (QBR prep, a scorecard).
When to ReAct: exploratory questions where the path is unknown.

Run:
    ../1.KPI_Narrator/.venv/bin/python 5.planning.py
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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


# ── Tools (plain functions; the planner references them by name) ────────────
def get_commitment(commitment_id: str) -> dict:
    data = {"C-09": {"promise": "SDK funnel tracking by August 2026", "owner": "Product"}}
    return data.get(commitment_id, {"error": "not found"})


def get_kpi(metric_id: str) -> dict:
    db = {"D-110": {"name": "SDK activations", "value": 12000, "prior": 24000, "target": 26000}}
    return db.get(metric_id, {"error": "not found"})


def rag_search(query: str) -> list:
    docs = [
        {"text": "Q2 QBR: promised SDK funnel visibility by August", "source_id": "qbr_q2"},
        {"text": "SDK rollout paused in 2 regions", "source_id": "ops_w25"},
    ]
    words = query.lower().split()
    return [d for d in docs if any(w in d["text"].lower() for w in words)]


TOOL_MAP = {"get_commitment": get_commitment, "get_kpi": get_kpi, "rag_search": rag_search}
TOOL_DOCS = (
    "get_commitment(commitment_id: str)  e.g. C-09\n"
    "get_kpi(metric_id: str)             e.g. D-110\n"
    "rag_search(query: str)              free-text search of QBR/ops docs"
)


def _extract_json(text: str) -> str:
    """Pull a JSON array out of the model's reply (strip ``` fences if present)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    start, end = body.find("["), body.rfind("]")
    return body[start:end + 1] if start != -1 and end != -1 else body


def make_plan(question: str) -> list:
    """Ask the LLM for an ordered list of tool calls as JSON."""
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a planner. Given a task and the available tools, output an "
            "ordered plan as a JSON array of steps. Each step is "
            '{"tool": <name>, "args": {<arg>: <value>}}. '
            "Output ONLY the JSON array, no prose.\n\nTOOLS:\n" + TOOL_DOCS
        )),
        HumanMessage(content=question),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return json.loads(_extract_json(text))


def execute_plan(plan: list) -> list:
    """Run each planned tool call, collecting observations."""
    observations = []
    for i, step in enumerate(plan):
        tool, args = step["tool"], step.get("args", {})
        result = TOOL_MAP[tool](**args) if tool in TOOL_MAP else {"error": f"unknown tool {tool}"}
        print(f"  step {i+1}: {tool}({args}) -> {result}")
        observations.append({"step": i + 1, "tool": tool, "args": args, "result": result})
    return observations


def synthesise(question: str, observations: list) -> str:
    resp = llm.invoke([
        SystemMessage(content=(
            "Write the final answer using ONLY the observations. "
            "Cite metric IDs and source_id. Never invent numbers."
        )),
        HumanMessage(content=(
            f"QUESTION:\n{question}\n\nOBSERVATIONS:\n{json.dumps(observations, indent=2)}"
        )),
    ])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def main() -> None:
    question = "Prepare a QBR note on the SDK funnel promise (C-09) vs delivery (D-110)."
    print("Q:", question, "\n")

    print("PLAN:")
    plan = make_plan(question)
    for i, step in enumerate(plan):
        print(f"  {i+1}. {step['tool']}({step.get('args', {})})")

    print("\nEXECUTE:")
    observations = execute_plan(plan)

    print("\nANSWER:")
    print(synthesise(question, observations))


if __name__ == "__main__":
    main()


"""
Q: Prepare a QBR note on the SDK funnel promise (C-09) vs delivery (D-110). 

PLAN:
  1. get_commitment({'commitment_id': 'C-09'})
  2. get_kpi({'metric_id': 'D-110'})

EXECUTE:
  step 1: get_commitment({'commitment_id': 'C-09'}) -> {'promise': 'SDK funnel tracking by August 2026', 'owner': 'Product'}
  step 2: get_kpi({'metric_id': 'D-110'}) -> {'name': 'SDK activations', 'value': 12000, 'prior': 24000, 'target': 26000}

ANSWER:
**QBR Note: SDK Funnel Promise (C-09) vs. Delivery (D-110)**

**Commitment (C-09):**
The Product team has committed to "SDK funnel tracking by August 2026". This commitment focuses on establishing the necessary tracking capabilities within the SDK funnel.

**Delivery (D-110 - SDK activations):**
Current SDK activations (D-110) are at 12,000. This represents a significant decrease from the prior period's 24,000 and is substantially below the target of 26,000.

**Analysis:**
The commitment (C-09) to implement "SDK funnel tracking" by August 2026 is a future-dated initiative aimed at providing visibility into the SDK funnel. Concurrently, the current performance of "SDK activations" (D-110) shows a concerning decline, falling short of both prior performance and target. The delivery of robust SDK funnel tracking (C-09) is critical to gain insights into the underlying causes of the performance observed in metrics like SDK activations (D-110) and to inform strategies for improvement.

"""