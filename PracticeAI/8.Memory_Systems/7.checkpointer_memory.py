"""
Example G — LangGraph checkpointer (workflow memory)
====================================================

A checkpointer saves the entire graph state, allowing the workflow to pause and later continue from exactly where it left off.

For agent orchestration (Topic 7), "memory" means the WORKFLOW's state persisted
between runs. A checkpointer saves the graph's state keyed by a `thread_id`, so:

  - re-invoking with the SAME thread_id CONTINUES from the saved state
  - a DIFFERENT thread_id starts fresh

This is what makes HITL possible (Topic 7, Example 5): the graph can pause for
hours and resume on the same thread. Here we show the memory aspect plainly — a
running log that accumulates across separate `invoke()` calls — with no LLM, so
it's free and deterministic.


With a Checkpointer

Instead:

Today

START

↓

Probing

↓

Adoption

↓

Narrative

↓

Save State

↓

Waiting...

Tomorrow

Load State

↓

Continue from Human Approval

No need to rerun the earlier nodes.


Setup:
    ../1.KPI_Narrator/.venv/bin/pip install langgraph
    ../1.KPI_Narrator/.venv/bin/python 7.checkpointer_memory.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

def last_two(old, new):
    return (old + new)[-2:]

def replace_reducer(old, new):
    return new+old


# `log` uses an `operator.add` reducer, so each node's returned list is APPENDED
# to the saved state rather than replacing it — that's how memory accumulates.
class SessionState(TypedDict):
    log: Annotated[list, operator.add]

# class SessionState(TypedDict):
#     log: Annotated[list, last_two]

def record_node(state: SessionState) -> dict:
    print("State received:", state)
    # In a real graph this would be an agent step; here it just records a turn.
    return {"log": [f"turn recorded (now {len(state['log']) + 1} total)"]}


def build_app():
    g = StateGraph(SessionState)
    g.add_node("record", record_node)
    g.add_edge(START, "record")
    g.add_edge("record", END)
    # The checkpointer is what gives the graph memory across invoke() calls.
    return g.compile(checkpointer=MemorySaver())
    # return g.compile()


def main() -> None:
    app = build_app()

    session_a = {"configurable": {"thread_id": "ioh-session-42"}}
    session_b = {"configurable": {"thread_id": "other-session-99"}}

    # Three separate calls on the SAME thread — state carries over each time.
    print("Session A (thread ioh-session-42):")
    for _ in range(3):
        state = app.invoke({"log": []}, session_a)
        print("  log:", state["log"])

    # A DIFFERENT thread is independent — starts empty.
    print("\nSession B (thread other-session-99):")
    state_b = app.invoke({"log": []}, session_b)
    print("  log:", state_b["log"])

    print(
        "\nSession A's log grew to 3 entries across 3 separate invoke() calls —\n"
        "the checkpointer remembered state between them. Session B, a different\n"
        "thread_id, started fresh. thread_id = session key; checkpointer = memory."
    )


if __name__ == "__main__":
    main()

"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 7.checkpointer_memory.py 
Session A (thread ioh-session-42):
  log: ['turn recorded (now 1 total)']
  log: ['turn recorded (now 1 total)', 'turn recorded (now 2 total)']
  log: ['turn recorded (now 1 total)', 'turn recorded (now 2 total)', 'turn recorded (now 3 total)']

Session B (thread other-session-99):
  log: ['turn recorded (now 1 total)']

Session A's log grew to 3 entries across 3 separate invoke() calls —
the checkpointer remembered state between them. Session B, a different
thread_id, started fresh. thread_id = session key; checkpointer = memory.

without checkpointer:

venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 7.checkpointer_memory.py 
Session A (thread ioh-session-42):
  log: ['turn recorded (now 1 total)']
  log: ['turn recorded (now 1 total)']
  log: ['turn recorded (now 1 total)']

Session B (thread other-session-99):
  log: ['turn recorded (now 1 total)']

Session A's log grew to 3 entries across 3 separate invoke() calls —
the checkpointer remembered state between them. Session B, a different
thread_id, started fresh. thread_id = session key; checkpointer = memory.

for last_two:
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 7.checkpointer_memory.py 
Session A (thread ioh-session-42):
State received: {'log': []}
  log: ['turn recorded (now 1 total)']
State received: {'log': ['turn recorded (now 1 total)']}
  log: ['turn recorded (now 1 total)', 'turn recorded (now 2 total)']
State received: {'log': ['turn recorded (now 1 total)', 'turn recorded (now 2 total)']}
  log: ['turn recorded (now 2 total)', 'turn recorded (now 3 total)']

Session B (thread other-session-99):
State received: {'log': []}
  log: ['turn recorded (now 1 total)']

Session A's log grew to 3 entries across 3 separate invoke() calls —
the checkpointer remembered state between them. Session B, a different
thread_id, started fresh. thread_id = session key; checkpointer = memory.
"""