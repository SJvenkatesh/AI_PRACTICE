"""
Example D — Prebuilt ReAct agent (LangGraph)
============================================

Example C hand-wrote the agent loop. In practice you rarely do — LangGraph's
`create_react_agent` gives you a production-grade ReAct loop (tool calling,
message state, step limits) in one call. You supply the model + tools; it runs
the loop.

LLM

↓

Tool requested?

↓

Execute tool

↓

Add ToolMessage

↓

LLM

↓

Tool requested?

↓

Execute tool

↓

...

↓

Final answer


It gives you a production-ready implementation of the ReAct loop, including:

Tool calling: Executes the tool requested by the model.
Message state: Maintains the conversation history (user, assistant, and tool messages).
Loop control: Continues calling the model until it stops requesting tools.
Step limits: Prevents infinite loops by capping the number of iterations.
Error handling: Helps manage tool execution failures gracefully.

These are all things you'd otherwise need to implement yourself.


The reference used ChatOpenAI; this uses free Gemini. Same `@tool` functions.

| Example C                         | Example D                                  |
| --------------------------------- | ------------------------------------------ |
| You implement the ReAct loop      | LangGraph provides the ReAct loop          |
| You inspect `response.tool_calls` | LangGraph handles tool calls automatically |
| You create `ToolMessage` objects  | LangGraph creates and manages them         |
| You manage conversation state     | LangGraph maintains message state          |
| More code, more control           | Less code, easier to build and maintain    |


Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langgraph
    ../1.KPI_Narrator/.venv/bin/python 4.langchain_agent.py
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

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


@tool
def get_kpi(metric_id: str) -> str:
    """Fetch a KPI by metric ID (e.g. D-110, D-150)."""
    db = {
        "D-110": {"name": "SDK activations", "value": 12000, "prior": 24000},
        "D-150": {"name": "ARPU IDR", "value": 44000, "prior": 42000},
    }
    return json.dumps(db.get(metric_id, {"error": "not found"}))


@tool
def search_docs(query: str) -> str:
    """Search QBR and ops documents for relevant snippets."""
    docs = [
        {"text": "Week 25: SDK activation fell from 24k to 12k (KPI D-110)", "source_id": "D-110"},
        {"text": "SDK rollout paused in 2 regions, reducing activation", "source_id": "ops_w25"},
        {"text": "Q2 commitment: SDK funnel by August", "source_id": "qbr_q2"},
    ]
    words = query.lower().split()
    hits = [d for d in docs if any(w in d["text"].lower() for w in words)]
    return json.dumps(hits)


def main() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # One call builds the whole ReAct loop. `prompt` is the system instruction.
    agent = create_react_agent(
        llm,
        tools=[get_kpi, search_docs],
        prompt=(
            "You are Wise Albert's analysis agent. Use tools to gather evidence. "
            "Cite metric IDs and source_id. Never invent numbers."
        ),
    )

    result = agent.invoke({
        "messages": [("user", "Summarize the activation drop with evidence.")]
    })

    # The full message list includes tool calls + results; the last is the answer.
    # Gemini may return content as a list of blocks — join the text parts.
    final = result["messages"][-1].content
    if not isinstance(final, str):
        final = "".join(
            b.get("text", "") for b in final if isinstance(b, dict) and b.get("type") == "text"
        )
    print("A:", final)
    print(
        "\nSame ReAct loop as Example C, but LangGraph manages the state machine,\n"
        "tool dispatch, and stop condition for you — less code, fewer bugs."
    )


if __name__ == "__main__":
    main()
