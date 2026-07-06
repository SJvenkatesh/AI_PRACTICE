"""
Example E — Human-in-the-loop with a real pause/resume (LangGraph interrupt)
============================================================================

The HITL gate in Example 3 just returned approved=True/False in code. Real HITL
literally PAUSES the workflow, surfaces the draft to a human (in a UI), and
RESUMES once they click Approve/Edit — possibly minutes later, in a different
process. LangGraph does this with:

  - `interrupt(payload)` inside a node: stops the graph and returns the payload
    to the caller.
  - a `checkpointer` (MemorySaver here): saves state so the graph can resume.
  - `Command(resume=<human decision>)`: re-invoke to continue from the interrupt.

This file mocks the draft (no LLM call) to keep the focus on the pause/resume
mechanism — and to run for free. In production the draft comes from the
narrative node (Example 3).

Rule of thumb — when is HITL required?
    Internal draft for CS     → optional
    Email to IOH              → required
    QBR PDF export            → required
    CEO Console tile          → often auto, with a confidence badge
    Anomaly alert to Slack    → auto if confidence > threshold

Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langgraph
    ../1.KPI_Narrator/.venv/bin/python 5.hitl_interrupt.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    question: str
    draft_answer: str
    approved: bool


def draft_node(state: State) -> dict:
    # Mocked draft (would come from the narrative LLM in production).
    return {"draft_answer": (
        "Activation fell 50% (D-110); commitment C-09 (SDK funnel) is at risk."
    )}


def hitl_node(state: State) -> dict:
    # PAUSE here. The payload is surfaced to the human/UI. Execution stops until
    # the caller resumes with Command(resume=...).
    decision = interrupt({
        "action": "approve_or_edit",
        "draft": state["draft_answer"],
    })
    # This line runs only AFTER resume. `decision` is whatever we resumed with.
    if decision.get("approved"):
        return {"draft_answer": decision.get("edited_text", state["draft_answer"]),
                "approved": True}
    return {"approved": False}


def build_app():
    g = StateGraph(State)
    g.add_node("draft", draft_node)
    g.add_node("hitl", hitl_node)
    g.add_edge(START, "draft")
    g.add_edge("draft", "hitl")
    g.add_edge("hitl", END)
    # A checkpointer is REQUIRED for interrupt/resume to work.
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    app = build_app()
    # thread_id ties the paused run to its saved state so we can resume it.
    config = {"configurable": {"thread_id": "qbr-demo-1"}}

    # 1) First invoke runs draft -> hits interrupt in hitl -> PAUSES.
    result = app.invoke({"question": "SDK funnel status for the QBR?"}, config)
    pause = result["__interrupt__"][0].value
    print("--- PAUSED for human review ---")
    print("draft surfaced to reviewer:", pause["draft"])

    # 2) A human reviews and edits. We resume with their decision.
    human_decision = {
        "approved": True,
        "edited_text": "Activation −50% (D-110); C-09 SDK-funnel commitment at risk. Escalating.",
    }
    print("\n--- human approved (with an edit); resuming ---")
    final = app.invoke(Command(resume=human_decision), config)

    print("\napproved:", final["approved"])
    print("final answer:", final["draft_answer"])
    print(
        "\nThe graph genuinely stopped at the gate and only continued after we fed\n"
        "the human's decision back in. The checkpointer held the state in between —\n"
        "in production that pause could span minutes and a separate approval UI."
    )


if __name__ == "__main__":
    main()
