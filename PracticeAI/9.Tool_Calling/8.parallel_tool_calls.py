"""
Example H — Parallel tool calls
===============================

A single assistant turn can request MULTIPLE tools at once, e.g.
`[get_kpi(D-110), get_kpi(D-150), rag_search(...)]`. Those calls are independent,
so run them concurrently instead of one-by-one — the turn finishes in about the
time of the slowest single tool.

    tool_calls = [get_kpi(D-110), get_kpi(D-150), rag_search(...)]
        → ThreadPoolExecutor runs all three at once
        → collect results (keep tool_call_id order for the model)

The tool calls here are mocked (each sleeps ~1s) so this runs free and shows the
timing win. Reminder: return one `tool` message per call, each with its
`tool_call_id` (Example C).

Run:
    ../1.KPI_Narrator/.venv/bin/python 8.parallel_tool_calls.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor


def get_kpi(metric_id: str) -> dict:
    time.sleep(1.0)  # simulate a DB/API round-trip
    db = {"D-110": {"value": 12000, "prior": 24000}, "D-150": {"value": 44000, "prior": 42000}}
    return {"metric_id": metric_id, **db.get(metric_id, {})}


def rag_search(query: str) -> list:
    time.sleep(1.0)
    return [{"text": "SDK rollout paused in 2 regions", "source_id": "ops_w25"}]


TOOL_FUNCTIONS = {"get_kpi": get_kpi, "rag_search": rag_search}


def execute_tool(name: str, args: dict) -> str:
    try:
        return json.dumps(TOOL_FUNCTIONS[name](**args))
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}"})


# What one assistant turn requested (id mirrors the model's tool_call ids).
TOOL_CALLS = [
    {"id": "call_1", "name": "get_kpi", "args": {"metric_id": "D-110"}},
    {"id": "call_2", "name": "get_kpi", "args": {"metric_id": "D-150"}},
    {"id": "call_3", "name": "rag_search", "args": {"query": "SDK activation drop"}},
]


def run_sequential(calls: list) -> dict:
    return {c["id"]: execute_tool(c["name"], c["args"]) for c in calls}


def run_parallel(calls: list) -> dict:
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {pool.submit(execute_tool, c["name"], c["args"]): c["id"] for c in calls}
        # Map each result back to its tool_call_id so the model can pair them.
        return {futures[f]: f.result() for f in futures}


def main() -> None:
    print(f"one turn requested {len(TOOL_CALLS)} tools\n")

    t0 = time.time()
    run_sequential(TOOL_CALLS)
    seq = time.time() - t0

    t0 = time.time()
    results = run_parallel(TOOL_CALLS)
    par = time.time() - t0

    for call_id, result in results.items():
        print(f"  {call_id}: {result}")
    print(f"\nsequential: {seq:.1f}s   parallel: {par:.1f}s")
    print(
        "\n3 tools x ~1s = ~3s one-by-one, but ~1s in parallel (the slowest single\n"
        "call). Each result is keyed by its tool_call_id so you can return one\n"
        "matching `tool` message per call, in any order."
    )


if __name__ == "__main__":
    main()

"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/9.Tool_Calling$ ../1.KPI_Narrator/.venv/bin/python 8.parallel_tool_calls.py 
one turn requested 3 tools

  call_1: {"metric_id": "D-110", "value": 12000, "prior": 24000}
  call_2: {"metric_id": "D-150", "value": 44000, "prior": 42000}
  call_3: [{"text": "SDK rollout paused in 2 regions", "source_id": "ops_w25"}]

sequential: 3.0s   parallel: 1.0s

3 tools x ~1s = ~3s one-by-one, but ~1s in parallel (the slowest single
call). Each result is keyed by its tool_call_id so you can return one
matching `tool` message per call, in any order.

"""