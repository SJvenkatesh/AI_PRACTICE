"""
Example G — Debugging agent workflows with traces
=================================================

When a pipeline gives a wrong answer, a trace tells you WHICH span went wrong, so
you fix the right thing instead of guessing:

  RETRIEVAL span  → rag_search returned "holiday_policy" not "qbr_q2"
                    FIX: hybrid search / better chunking
  TOOL span       → get_kpi called with D-999 (wrong ID)
                    FIX: better tool description
  LLM span        → correct context but the answer invents a number
                    FIX: stronger system prompt / reflection step
  ORCHESTRATION   → adoption_agent never dispatched
                    FIX: intent classifier missed "promise_check"

This reads a JSONL trace log and prints each span with ✅ / ❌. It writes its own
demo traces (one healthy, one with a failing retrieval span) so you can see both
paths. It can also read the sibling `traces.jsonl` produced by
`1.TraceStructure..py` — pass that path to debug_trace().

No LLM. Run:
    ../1.KPI_Narrator/.venv/bin/python 6.debug_trace.py
"""

import json
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
DEMO_LOG = LOG_DIR / "demo_traces.jsonl"

# Two demo traces: one clean, one where retrieval fetched the wrong doc.
DEMO_TRACES = [
    {
        "trace_id": "trace-healthy-001",
        "question": "Why did D-110 drop?",
        "spans": [
            {"type": "llm", "name": "classify_intent", "latency_ms": 320, "error": None},
            {"type": "tool", "name": "get_kpi", "latency_ms": 45, "error": None},
            {"type": "retrieval", "name": "rag_search", "latency_ms": 210, "error": None},
            {"type": "llm", "name": "narrative_llm", "latency_ms": 1800, "error": None},
        ],
    },
    {
        "trace_id": "trace-buggy-002",
        "question": "Which commitment is at risk?",
        "spans": [
            {"type": "llm", "name": "classify_intent", "latency_ms": 300, "error": None},
            {"type": "retrieval", "name": "rag_search", "latency_ms": 190,
             "error": "returned 'holiday_policy' (irrelevant) instead of 'qbr_q2'"},
            {"type": "llm", "name": "narrative_llm", "latency_ms": 1700,
             "error": "cited the wrong commitment (bad context from retrieval)"},
        ],
    },
]


def write_demo_traces() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with open(DEMO_LOG, "w") as f:
        for t in DEMO_TRACES:
            f.write(json.dumps(t) + "\n")


def debug_trace(trace_id: str, log_file: Path = DEMO_LOG) -> None:
    with open(log_file) as f:
        for line in f:
            trace = json.loads(line)
            if trace["trace_id"] == trace_id:
                print(f"Question: {trace['question']}")
                for span in trace["spans"]:
                    status = "❌" if span.get("error") else "✅"
                    print(f"  {status} [{span['type']}] {span['name']} — {span['latency_ms']}ms")
                    if span.get("error"):
                        print(f"       error: {span['error']}")
                return
    print(f"Trace {trace_id} not found in {log_file}")


def main() -> None:
    write_demo_traces()

    print("=== Healthy trace ===")
    debug_trace("trace-healthy-001")

    print("\n=== Buggy trace (the one you'd investigate) ===")
    debug_trace("trace-buggy-002")

    print(
        "\nThe ❌ spans localise the failure: retrieval fetched the wrong doc, which\n"
        "then poisoned the narrative. You'd fix retrieval (hybrid search / chunking)\n"
        "— not the prompt. Traces turn 'the answer is wrong' into a precise bug."
    )


if __name__ == "__main__":
    main()
