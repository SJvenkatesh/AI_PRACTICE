"""
Example G — Wise Albert Adoption Agent (realistic)
==================================================

A fuller agent that ties the topic together:

  GOAL:      produce a "promise vs delivery" scorecard.
  TOOLS:     get_sdk_funnel, get_commitments, rag_search  (read-only, safe)
             update_promise_status                        (WRITE — state-changing)
  GUARDRAILS:
     • Numbers only from tools (system prompt + tools are the only number source).
     • Human-in-the-loop (HITL): the WRITE tool cannot run without approval.
  ARTIFACT:  a Promise Scorecard written to ./promise_scorecard.md

The perceive → reason → act → produce shape:
  perceive = read tools, reason = the LLM's tool choices, act = the write tool
  (gated), produce = the scorecard file.

Run:
    ../1.KPI_Narrator/.venv/bin/python 7.adoption_agent.py
    # WRITE actions auto-approve in a non-interactive run (with a notice); in a
    # real terminal you'd be prompted y/N.
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
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

# A tiny in-memory "database" the write tool mutates.
PROMISE_DB = {"C-09": {"promise": "SDK funnel tracking by August 2026", "status": "unknown"}}


# ── READ tools (safe, auto-run) ─────────────────────────────────────────────
@tool
def get_sdk_funnel(stage: str) -> dict:
    """Get SDK funnel numbers for a stage: 'activation' or 'install'."""
    funnel = {
        "install": {"value": 30000, "prior": 31000},
        "activation": {"value": 12000, "prior": 24000, "target": 26000},
    }
    return funnel.get(stage, {"error": "unknown stage"})


@tool
def get_commitments(quarter: str) -> list:
    """List commitments for a quarter, e.g. 'Q2-2026'."""
    return [{"commitment_id": "C-09", "promise": "SDK funnel tracking by August 2026"}]


@tool
def rag_search(query: str) -> list:
    """Search QBR/ops history for context."""
    docs = [
        {"text": "SDK rollout paused in 2 regions", "source_id": "ops_w25"},
        {"text": "Q2 commitment: SDK funnel by August", "source_id": "qbr_q2"},
    ]
    words = query.lower().split()
    return [d for d in docs if any(w in d["text"].lower() for w in words)]


# ── WRITE tool (state-changing → HITL-gated) ────────────────────────────────
@tool
def update_promise_status(commitment_id: str, status: str) -> dict:
    """Set a commitment's status to on_track | at_risk | missed. STATE-CHANGING."""
    if commitment_id not in PROMISE_DB:
        return {"error": "unknown commitment"}
    PROMISE_DB[commitment_id]["status"] = status
    return {"ok": True, "commitment_id": commitment_id, "status": status}


READ_TOOLS = {"get_sdk_funnel", "get_commitments", "rag_search"}
WRITE_TOOLS = {"update_promise_status"}  # require human approval
TOOLS = [get_sdk_funnel, get_commitments, rag_search, update_promise_status]
TOOL_MAP = {t.name: t for t in TOOLS}

# flash-lite has a higher free-tier request cap than flash — better for an agent
# that makes several LLM calls per run (each tool round-trip is a request).
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2).bind_tools(TOOLS)


def _text(content) -> str:
    """Join Gemini content blocks (thinking + text) into a plain string."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "".join(parts).strip()


def hitl_approve(tool_name: str, args: dict) -> bool:
    """Human-in-the-loop gate for state-changing actions."""
    print(f"  [HITL GATE] agent wants to call {tool_name}({args})")
    if not sys.stdin.isatty():
        # Non-interactive (e.g. this demo): auto-approve but say so loudly.
        print("             non-interactive run -> auto-approving for demo")
        return True
    return input("             approve? [y/N] ").strip().lower() == "y"


def run_agent(goal: str, max_steps: int = 6) -> str:
    messages = [
        SystemMessage(content=(
            "You are Wise Albert's adoption agent. Build a promise-vs-delivery "
            "scorecard. Use tools for ALL numbers — never invent them. When you "
            "have gathered evidence, set the commitment status with "
            "update_promise_status, then write a short scorecard citing metric "
            "IDs and source_id."
        )),
        HumanMessage(content=goal),
    ]

    for step in range(max_steps):
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return _text(ai.content)

        for tc in ai.tool_calls:
            name, args = tc["name"], tc["args"]
            if name in WRITE_TOOLS and not hitl_approve(name, args):
                result = {"error": "denied by human reviewer"}
            else:
                if name in READ_TOOLS:
                    print(f"  [step {step}] READ: {name}({args})")
                result = TOOL_MAP[name].invoke(args)
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    return "Max steps reached — incomplete."


def main() -> None:
    goal = "Produce the SDK funnel promise-vs-delivery scorecard for Q2-2026."
    print("GOAL:", goal, "\n")

    scorecard = run_agent(goal)

    print("\n=== PROMISE SCORECARD ===")
    print(scorecard)
    print(f"\nfinal promise DB state: {PROMISE_DB}")

    # ARTIFACT: persist the scorecard (perceive/reason/act -> produce).
    out = Path(__file__).parent / "promise_scorecard.md"
    out.write_text(f"# Promise Scorecard (Q2-2026)\n\n{scorecard}\n")
    print(f"\nartifact written: {out.name}")


if __name__ == "__main__":
    main()
