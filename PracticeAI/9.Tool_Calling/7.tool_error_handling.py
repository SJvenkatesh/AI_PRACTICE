"""
Example G — Handling tool errors gracefully
============================================

Tools fail: unknown name, bad data, timeouts, exceptions. If a tool raises and
crashes your loop, the agent dies. Instead, catch everything and return a short
error STRING the model can read — then the LLM can tell the user
"I couldn't fetch D-110 right now" and try another approach.

Two rules:
  - Return errors as data (JSON), don't raise out of the loop.
  - Don't leak stack traces / internals to the model in production — a terse
    message is enough for it to react to.

No LLM — we call execute_tool() on several failure modes to show the handling.

Run:
    ../1.KPI_Narrator/.venv/bin/python 7.tool_error_handling.py
"""

import json


def get_kpi(metric_id: str) -> dict:
    db = {"D-110": {"value": 12000, "prior": 24000}}
    if metric_id not in db:
        return {"error": f"Metric {metric_id} not found"}   # expected "miss"
    return {"metric_id": metric_id, **db[metric_id]}


def slow_tool(query: str) -> dict:
    raise TimeoutError("upstream took too long")             # simulated timeout


def buggy_tool(x: int) -> dict:
    return {"result": 10 / x}                                # ZeroDivisionError if x=0


TOOL_FUNCTIONS = {"get_kpi": get_kpi, "slow_tool": slow_tool, "buggy_tool": buggy_tool}


def execute_tool(name: str, args: dict) -> str:
    """Run a tool, always returning a JSON string (never raising)."""
    try:
        if name not in TOOL_FUNCTIONS:
            return json.dumps({"error": f"Unknown tool: {name}"})
        result = TOOL_FUNCTIONS[name](**args)
        return json.dumps(result)              # includes the tool's own {"error": ...}
    except TimeoutError:
        return json.dumps({"error": "Tool timed out. Try a narrower query."})
    except Exception as e:
        # Log e with the full trace server-side; return only the type to the model.
        return json.dumps({"error": f"Tool failed: {type(e).__name__}"})


def main() -> None:
    cases = [
        ("get_kpi", {"metric_id": "D-110"}),    # success
        ("get_kpi", {"metric_id": "D-999"}),    # tool's own soft error (not found)
        ("does_not_exist", {}),                 # unknown tool
        ("slow_tool", {"query": "x"}),          # TimeoutError
        ("buggy_tool", {"x": 0}),               # unexpected exception (ZeroDivision)
    ]
    for name, args in cases:
        print(f"{name}({args})")
        print(f"  -> {execute_tool(name, args)}\n")

    print(
        "Every call returned clean JSON — nothing raised out of the loop. The model\n"
        "receives each error as data and can respond ('data source unavailable,\n"
        "let me try…') instead of the program crashing on a stack trace."
    )


if __name__ == "__main__":
    main()


"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/9.Tool_Calling$ ../1.KPI_Narrator/.venv/bin/python 7.tool_error_handling.py
get_kpi({'metric_id': 'D-110'})
  -> {"metric_id": "D-110", "value": 12000, "prior": 24000}

get_kpi({'metric_id': 'D-999'})
  -> {"error": "Metric D-999 not found"}

does_not_exist({})
  -> {"error": "Unknown tool: does_not_exist"}

slow_tool({'query': 'x'})
  -> {"error": "Tool timed out. Try a narrower query."}

buggy_tool({'x': 0})
  -> {"error": "Tool failed: ZeroDivisionError"}

Every call returned clean JSON — nothing raised out of the loop. The model
receives each error as data and can respond ('data source unavailable,
let me try…') instead of the program crashing on a stack trace.
"""