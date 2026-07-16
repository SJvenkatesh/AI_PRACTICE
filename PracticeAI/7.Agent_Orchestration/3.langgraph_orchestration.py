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


                User Question
                      │
                      ▼
               classify_node
                      │
          (Which agents are needed?)
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    probing      adoption      prediction
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                evidence_bus
                      ▼
               narrative_node
                      ▼
                  hitl_node
                      │
             Approved? │ No
                  Yes  ▼
                      revise
                        │
                        └──────────────► narrative


Step 1 — classify_node

This is the intent router.

Input:

Why did activation drop?

Prompt:

Determine which agents are required.

Possible agents:

- probing
- adoption
- prediction
- personal_intel

Return JSON.

Suppose Gemini returns:

{
  "agents": [
    "probing",
    "adoption"
  ]
}

Now the planner knows:

Run

probing
adoption

Skip

prediction
Step 2 — Agent Nodes

Each node checks

Am I selected?

Suppose

selected = [
    "probing",
    "adoption"
]
Probing Node
if "probing" not in selected:
    return {}

False

So it executes

return {
    "probing_result": ...
}
Adoption Node

Also selected

Returns

{
    "adoption_result": ...
}
Prediction Node

Checks

if "prediction" not in selected:

True

So it returns

{}

Nothing happens.

This is what the sentence means:

run ONLY if the planner chose them (else return {})

The node still exists in the graph, but it doesn't do any work if it wasn't selected.

Step 3 — evidence_bus

Now the state contains

{
    "probing_result": {...},
    "adoption_result": {...}
}

Prediction returned

{}

So nothing is added.

The evidence bus collects everything into one package.

Example

evidence = {
    "probing": state["probing_result"],
    "adoption": state["adoption_result"]
}

Now downstream agents don't need to know where evidence came from.

They simply receive

evidence
Step 4 — narrative_node

Narrative receives

{
    "probing": {...},
    "adoption": {...}
}

Prompt

Use ONLY this evidence.

Write an executive summary.

Maximum 4 sentences.

Gemini writes

Activation declined 50%.

The August commitment is at risk.

Step 5 — hitl_node

HITL means

Human In The Loop

Suppose this report is going to a customer.

Instead of automatically sending it

LangGraph pauses.

Narrative

↓

Manager Review

Manager says

Looks good.

Approve.

Workflow ends.

Suppose manager says

Sentence 2 is incorrect.

Then

Narrative

↓

Human

↓

Revise

↓

Narrative

This is the loop shown here

             narrative

↓

HITL

↓

Revise

↓

Narrative
Why have an Evidence Bus?

Without it

Narrative might need

probing_result

adoption_result

prediction_result

...

As you add more agents, this becomes messy.

Instead

Narrative only receives

evidence

which is a single object.


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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


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
    print("\nResult:\n", result)
    print("\n OrchestratorState:\n", OrchestratorState)
    print(
        "\nOne graph did it all: classify → gather (only chosen agents) → merge →\n"
        "narrate → HITL gate. Change the question to a 'qbr_prep' one and the gate\n"
        "would withhold approval instead."
    )


if __name__ == "__main__":
    main()

"""
output:

venkatesh@venkatesh:~/WiseAlbert/PracticeAI/7.Agent_Orchestration$ ../1.KPI_Narrator/.venv/bin/python 3.langgraph_orchestration.py
Intents: ['kpi_anomaly']
Agents run: ['probing', 'prediction']
Approved (no human needed): True

Draft answer:
 Activation (D-110) has dropped sharply from 24,000 to 12,000, triggering a 'red' severity alert. Our forecast indicates a further decline to 11,000 over the next 7 days with 0.88 confidence. The root cause for this anomaly is not yet identified in the data. There is no information regarding C-09 available in the current evidence.

Result:
 {'question': 'Why did activation drop? Is C-09 at risk?', 'persona': 'cs_lead', 'intents': ['kpi_anomaly'], 'agents_to_run': ['probing', 'prediction'], 'probing_result': {'D-110': {'value': 12000, 'prior': 24000, 'severity': 'red'}}, 'adoption_result': {}, 'prediction_result': {'forecast_7d': 11000, 'confidence': 0.88}, 'evidence_package': {'probing': {'D-110': {'value': 12000, 'prior': 24000, 'severity': 'red'}}, 'adoption': {}, 'prediction': {'forecast_7d': 11000, 'confidence': 0.88}, 'intents': ['kpi_anomaly']}, 'draft_answer': "Activation (D-110) has dropped sharply from 24,000 to 12,000, triggering a 'red' severity alert. Our forecast indicates a further decline to 11,000 over the next 7 days with 0.88 confidence. The root cause for this anomaly is not yet identified in the data. There is no information regarding C-09 available in the current evidence.", 'approved': True}

 OrchestratorState:
 <class '__main__.OrchestratorState'>

One graph did it all: classify → gather (only chosen agents) → merge →
narrate → HITL gate. Change the question to a 'qbr_prep' one and the gate
would withhold approval instead.

"""