"""
Every user request gets one trace_id:
"""

import uuid
import time
import json
from datetime import datetime, timezone


def get_kpi(metric_id: str, weeks: int = 1) -> dict:
    #The docstring (the triple-quoted """...""" comment immediately below the function. LangChain inspects this function using Python's introspection.
    """Get live KPI values by metric ID (D-110, D-150). Never invent numbers."""
    db = {
        "D-110": {"name": "SDK activations", "value": 12000, "prior": 24000, "unit": "count"},
        "D-150": {"name": "ARPU", "value": 44000, "prior": 42000, "unit": "IDR"},
    }
    if metric_id not in db:
        return {"error": f"Metric {metric_id} not found"}
    return {"metric_id": metric_id, **db[metric_id], "weeks": weeks}


def rag_search(query: str, top_k: int = 3) -> list:
    """Search QBR notes and ops documents for context and commitments."""
    docs = [
        {"text": "D-110 SDK activation dropped after rollout paused in 2 regions", "source_id": "ops_w25"},
        {"text": "C-09 SDK funnel tracking status amber", "source_id": "qbr_q2"},
    ]
    words = query.lower().split()
    return [d for d in docs if any(w in d["text"].lower() for w in words)][:top_k]


def new_trace(user_question: str) -> dict:
    return {
        "trace_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "question": user_question,
        "spans": [],  # each step is a "span"
    }


def add_span(trace: dict, name: str, span_type: str, input_data: dict, output_data: dict, latency_ms: float, error: str = None):
    trace["spans"].append({
        "name": name,
        "type": span_type,       # "agent" | "tool" | "llm" | "retrieval"
        "input": input_data,
        "output": output_data,
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# Usage in pipeline
trace = new_trace("Why did D-110 drop?")

t0 = time.time()
kpi_result = get_kpi("D-110")
add_span(trace, "get_kpi", "tool", {"metric_id": "D-110"}, kpi_result, (time.time()-t0)*1000)

t0 = time.time()
docs = rag_search("activation drop")
add_span(trace, "rag_search", "retrieval", {"query": "activation drop"}, docs, (time.time()-t0)*1000)

# t0 = time.time()
# answer = call_llm(question, kpi_result, docs)
# add_span(trace, "narrative_llm", "llm", {"model": "gpt-4o-mini"}, {"answer": answer}, (time.time()-t0)*1000)

# Save trace
with open("traces.jsonl", "a") as f:
    f.write(json.dumps(trace) + "\n")