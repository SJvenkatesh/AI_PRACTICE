"""
Example C — LangGraph orchestration with state
==============================================

The full orchestration flow as one LangGraph, wiring together everything from
1.orchestration_flow.md:

    classify → probing → adoption → prediction → evidence_bus → narrative → hitl
                                                                     ▲        │
                                                                     └─ revise┘

- classify_node   : LLM picks intents, planner picks agents (Example B logic)
- *_node agents   : run ONLY if the planner chose them (else return {})
- evidence_bus    : merge agent JSON into one package (STEP 3)
- narrative_node  : LLM writes the answer from the evidence (STEP 4)
- hitl_node       : gate — external/QBR output needs a human (STEP 5)

Agents are mocked here (they return fixed JSON) so the demo is cheap — only
classify + narrative call the LLM. Example 4 shows running them in parallel.

Provider: free Gemini (flash-lite). Two LLM calls per run.

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langgraph
    ../1.KPI_Narrator/.venv/bin/python 3.langgraph_orchestration.py
"""

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Reuse the intent router from Example B (same folder). Its filename starts with
# a digit, so we load it via importlib rather than a normal import.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("intent_router", Path(__file__).parent / "2.intent_router.py")
intent_router = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(intent_router)
classify_intent, plan_agents = intent_router.classify_intent, intent_router.plan_agents

if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY not set. Put it in ../1.KPI_Narrator/.env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


class OrchestratorState(TypedDict):
    question: str
    persona: str
    intents: list
    agents_to_run: list
    probing_result: dict
    adoption_result: dict
    prediction_result: dict
    evidence_package: dict
    draft_answer: str
    approved: bool


def classify_node(state: OrchestratorState) -> dict:
    intents = classify_intent(state["question"], state["persona"])
    return {"intents": intents, "agents_to_run": plan_agents(intents)}


def probing_node(state: OrchestratorState) -> dict:
    if "probing" not in state["agents_to_run"]:
        return {"probing_result": {}}
    return {"probing_result": {"D-110": {"value": 12000, "prior": 24000, "severity": "red"}}}


def adoption_node(state: OrchestratorState) -> dict:
    if "adoption" not in state["agents_to_run"]:
        return {"adoption_result": {}}
    return {"adoption_result": {"C-09": {"status": "amber", "text": "SDK funnel by Aug"}}}


def prediction_node(state: OrchestratorState) -> dict:
    if "prediction" not in state["agents_to_run"]:
        return {"prediction_result": {}}
    return {"prediction_result": {"forecast_7d": 11000, "confidence": 0.88}}


def evidence_bus_node(state: OrchestratorState) -> dict:
    return {"evidence_package": {
        "probing": state.get("probing_result", {}),
        "adoption": state.get("adoption_result", {}),
        "prediction": state.get("prediction_result", {}),
        "intents": state.get("intents", []),
    }}


def narrative_node(state: OrchestratorState) -> dict:
    draft = llm.invoke([
        SystemMessage(content=(
            f"You write for persona '{state['persona']}'. Use ONLY the evidence. "
            "Cite metric/commitment IDs. Max 4 sentences."
        )),
        HumanMessage(content=(
            f"Evidence: {state['evidence_package']}\nQuestion: {state['question']}"
        )),
    ])
    return {"draft_answer": _text(draft.content)}


def hitl_node(state: OrchestratorState) -> dict:
    # External / QBR output requires a human; internal drafts auto-approve.
    is_external = any("qbr" in i for i in state.get("intents", []))
    return {"approved": not is_external}


def route_after_hitl(state: OrchestratorState) -> Literal["end", "revise"]:
    return "end" if state.get("approved") else "revise"


def build_app():
    g = StateGraph(OrchestratorState)
    for name, fn in [
        ("classify", classify_node), ("probing", probing_node),
        ("adoption", adoption_node), ("prediction", prediction_node),
        ("evidence_bus", evidence_bus_node), ("narrative", narrative_node),
        ("hitl", hitl_node),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    # Sequential for clarity (agents are mocked/cheap); Example 4 parallelises.
    g.add_edge("classify", "probing")
    g.add_edge("probing", "adoption")
    g.add_edge("adoption", "prediction")
    g.add_edge("prediction", "evidence_bus")
    g.add_edge("evidence_bus", "narrative")
    g.add_edge("narrative", "hitl")
    g.add_conditional_edges("hitl", route_after_hitl, {"end": END, "revise": "narrative"})
    return g.compile()


def main() -> None:
    app = build_app()
    result = app.invoke({
        "question": "Why did activation drop? Is C-09 at risk?",
        "persona": "cs_lead",
    })
    print("Intents:", result["intents"])
    print("Agents run:", result["agents_to_run"])
    print("Approved (no human needed):", result["approved"])
    print("\nDraft answer:\n", result["draft_answer"])
    print(
        "\nOne graph did it all: classify → gather (only chosen agents) → merge →\n"
        "narrate → HITL gate. Change the question to a 'qbr_prep' one and the gate\n"
        "would withhold approval instead."
    )


if __name__ == "__main__":
    main()
