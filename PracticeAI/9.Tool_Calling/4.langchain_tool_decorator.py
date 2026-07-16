"""
Example D — LangChain @tool decorator + prebuilt agent
======================================================

The @tool decorator turns a plain function into a tool the LLM can call — its
DOCSTRING becomes the description the model reads, and its type hints become the
parameter schema (Example A showed the generated schema). Hand those tools to
LangGraph's `create_react_agent` and the whole tool-calling loop (Example B) is
managed for you.

The lesson: write the docstring carefully — it's the tool's description to the
model, and it decides whether the tool gets picked at the right time.

Reference used ChatOpenAI; this uses free Gemini.

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langgraph
    ../1.KPI_Narrator/.venv/bin/python 4.langchain_tool_decorator.py
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
def get_kpi(metric_id: str, weeks: int = 1) -> str:
    """Fetch live KPI by metric ID (D-110, D-150). Use for all numeric metrics."""
    db = {
        "D-110": {"name": "SDK activations", "value": 12000, "prior": 24000},
        "D-150": {"name": "ARPU", "value": 44000, "prior": 42000},
    }
    return json.dumps(db.get(metric_id, {"error": "not found"}))


@tool
def search_docs(query: str) -> str:
    """Search internal QBR and ops documents. Use for context and commitments."""
    docs = [
        {"text": "C-09: SDK funnel tracking status amber", "source_id": "qbr_q2"},
        {"text": "D-110 SDK activation fell 24k to 12k", "source_id": "ops_w25"},
    ]
    words = query.lower().split()
    hits = [d for d in docs if any(w in d["text"].lower() for w in words)]
    return json.dumps(hits)


def main() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    agent = create_react_agent(
        llm,
        tools=[get_kpi, search_docs],
        prompt="You are Wise Albert. Cite metric_id and source_id. Never invent numbers.",
    )

    result = agent.invoke({
        "messages": [("user", "Is C-09 at risk given the D-110 trend?")]
    })

    final = result["messages"][-1].content
    if not isinstance(final, str):
        final = "".join(b.get("text", "") for b in final
                        if isinstance(b, dict) and b.get("type") == "text")
    print("A:", final)
    print(
        f"\n(the agent ran {len(result['messages'])} messages total: the question,\n"
        " its tool requests, your tool results, and the final answer — all managed\n"
        " by create_react_agent from just the two @tool functions.)"
    )


if __name__ == "__main__":
    main()

"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/9.Tool_Calling$ ../1.KPI_Narrator/.venv/bin/python 4.langchain_tool_decorator.py 
/home/venkatesh/WiseAlbert/PracticeAI/9.Tool_Calling/4.langchain_tool_decorator.py:69: LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
  agent = create_react_agent(
A: The D-110 trend shows a decrease in SDK activations, with the current value at 12,000 compared to a prior value of 24,000. This indicates a potential risk for C-09.

(the agent ran 4 messages total: the question,
 its tool requests, your tool results, and the final answer — all managed
 by create_react_agent from just the two @tool functions.)

 """