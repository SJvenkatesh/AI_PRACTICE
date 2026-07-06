"""
Example D — LangGraph multi-agent (state graph, parallel fan-out)
================================================================

Examples 1 & 2 wired agents together by hand. LangGraph models a multi-agent
workflow as a GRAPH: nodes are agents, edges are the flow, and a shared STATE
object is passed between them. This scales to routing, branching, and joins.

Shape here (an improvement on a plain sequential chain — the three specialists
run in PARALLEL, then a join into the narrative):

              ┌─► probing ───┐
    START ────┼─► adoption ──┼─► narrative ─► END
              └─► prediction ┘

LangGraph runs nodes with no dependency between them concurrently, and waits for
all edges into `narrative` before running it. Each agent writes its OWN key in
the shared state, so there's no write conflict.

Python
│
├── ThreadPoolExecutor
│      │
│      └── Runs functions concurrently
│
└── LangGraph
       │
       ├── Orchestrates AI workflows
       ├── Manages state
       ├── Routes between agents
       ├── Handles parallel execution
       └── May use Python concurrency internally

Provider: free Gemini (flash-lite).

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langgraph
    ../1.KPI_Narrator/.venv/bin/python 3.langgraph_supervisor.py
"""

import os
import sys
from pathlib import Path
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

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
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


# Shared state passed between nodes. Each agent fills in one field.
class WiseAlbertState(TypedDict):
    question: str
    probing_result: str
    adoption_result: str
    prediction_result: str
    final_answer: str


def probing_agent(state: WiseAlbertState) -> dict:
    r = llm.invoke(
        f"You are the Probing Agent. Analyze: {state['question']}. "
        "Return anomaly signals as JSON."
    )
    return {"probing_result": _text(r.content)}


def adoption_agent(state: WiseAlbertState) -> dict:
    r = llm.invoke(
        f"You are the Adoption Agent. Question: {state['question']}. "
        "Check promise/commitment risk. Return JSON."
    )
    return {"adoption_result": _text(r.content)}


def prediction_agent(state: WiseAlbertState) -> dict:
    r = llm.invoke(
        f"You are the Prediction Agent. Question: {state['question']}. "
        "Forecast the trend. Return JSON."
    )
    return {"prediction_result": _text(r.content)}


def narrative_agent(state: WiseAlbertState) -> dict:
    evidence = (
        f"Probing: {state.get('probing_result')}\n"
        f"Adoption: {state.get('adoption_result')}\n"
        f"Prediction: {state.get('prediction_result')}"
    )
    r = llm.invoke(
        f"You are the Narrative Agent for the CS Lead. Use ONLY:\n{evidence}\n"
        f"Cite sources. Max 4 sentences. Question: {state['question']}"
    )
    return {"final_answer": _text(r.content)}


def build_app():
    graph = StateGraph(WiseAlbertState)
    graph.add_node("probing", probing_agent)
    graph.add_node("adoption", adoption_agent)
    graph.add_node("prediction", prediction_agent)
    graph.add_node("narrative", narrative_agent)

    # Fan-out from START to all three specialists (they run in parallel)...
    graph.add_edge(START, "probing")
    graph.add_edge(START, "adoption")
    graph.add_edge(START, "prediction")
    # ...then join: narrative runs once all three have finished.
    graph.add_edge("probing", "narrative")
    graph.add_edge("adoption", "narrative")
    graph.add_edge("prediction", "narrative")
    graph.add_edge("narrative", END)
    return graph.compile()


def main() -> None:
    app = build_app()
    result = app.invoke({"question": "Why did activation drop? Any promise at risk?"})

    print("── probing_result ──\n", result["probing_result"], "\n")
    print("── adoption_result ──\n", result["adoption_result"], "\n")
    print("── prediction_result ──\n", result["prediction_result"], "\n")
    print("── final_answer (narrative) ──\n", result["final_answer"])
    print(
        "\nSame agents as Example 2, but LangGraph owns the wiring: shared state +\n"
        "graph nodes. The three specialists fanned out in parallel and joined at\n"
        "the narrative node — no manual thread pool or handoff code."
    )


if __name__ == "__main__":
    main()
