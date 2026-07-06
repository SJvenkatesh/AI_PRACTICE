"""
Example D — Parallel dispatch (production pattern)
==================================================

Example 3 ran the chosen agents one after another for clarity. In production the
orchestrator dispatches them in PARALLEL after classification, because they're
independent — this is what keeps the full pipeline fast (target: < 90 seconds).

    agents_to_run = ["probing", "adoption", "prediction"]
        → ThreadPoolExecutor submits all at once
        → collect results as they finish
        → hand the merged dict to the evidence bus

The agent runners here are mocked (each just sleeps briefly to simulate work),
so this file is free to run and shows the timing win without any LLM calls. In a
real graph you'd wrap this in a single "parallel_dispatch" node.

Run:
    ../1.KPI_Narrator/.venv/bin/python 4.parallel_dispatch.py
"""

import time
from concurrent.futures import ThreadPoolExecutor


# Mock agent runners. Each "takes ~1s" to simulate a tool call / LLM round-trip.
def probing_agent_run(q: str) -> dict:
    time.sleep(1.0)
    return {"D-110": {"value": 12000, "prior": 24000, "severity": "red"}}


def adoption_agent_run(q: str) -> dict:
    time.sleep(1.0)
    return {"C-09": {"status": "amber", "text": "SDK funnel by Aug"}}


def prediction_agent_run(q: str) -> dict:
    time.sleep(1.0)
    return {"forecast_7d": 11000, "confidence": 0.88}


AGENT_RUNNERS = {
    "probing": probing_agent_run,
    "adoption": adoption_agent_run,
    "prediction": prediction_agent_run,
}


def dispatch_parallel(agents: list, question: str) -> dict:
    """Run the chosen agents concurrently; return {agent_name: result}."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        futures = {pool.submit(AGENT_RUNNERS[a], question): a for a in agents}
        for f in futures:
            results[futures[f]] = f.result()
    return results


def dispatch_sequential(agents: list, question: str) -> dict:
    """Same work, one after another — for the timing comparison."""
    return {a: AGENT_RUNNERS[a](question) for a in agents}


def main() -> None:
    agents = ["probing", "adoption", "prediction"]  # from the planner (Example B)
    question = "Why did activation drop? Is C-09 at risk?"

    t0 = time.time()
    seq = dispatch_sequential(agents, question)
    seq_time = time.time() - t0

    t0 = time.time()
    par = dispatch_parallel(agents, question)
    par_time = time.time() - t0

    print("evidence (parallel):", par)
    print(f"\nsequential: {seq_time:.1f}s   parallel: {par_time:.1f}s")
    print(
        "\n3 agents x ~1s each = ~3s sequential, but ~1s in parallel (the slowest\n"
        "single agent). With real LLM/tool calls the gap is far larger — which is\n"
        "why parallel dispatch is essential to hit the <90s target."
    )


if __name__ == "__main__":
    main()
