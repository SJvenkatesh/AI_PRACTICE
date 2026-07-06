"""
Example C — Parallel agents (faster)
====================================

In the sequential handoff (Example 1), each agent waited for the previous one.
But Probing, Adoption, and Prediction don't depend on each other — they can all
look at the same data AT THE SAME TIME. Running them in parallel cuts wall-clock
time to roughly the slowest single agent instead of the sum of all three.

    ┌─ Probing ────┐
    ├─ Adoption ───┤  (run concurrently)
    └─ Prediction ─┘
           │
           ▼
      Narrative (merge all evidence, write once)

We use a ThreadPoolExecutor: I/O-bound LLM calls (waiting on the network) overlap
well with threads. Then one Narrative agent merges the gathered evidence.

Provider: free Gemini (flash-lite). Note: parallel calls hit the API at once, so
on a tight free tier you may see a rate-limit — lower max_workers if so.

                ThreadPoolExecutor
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
 run_probing()   run_adoption()  run_prediction()
        │              │              │
        ▼              ▼              ▼
   Future1        Future2        Future3
        │              │              │
        └──────┬───────┴───────┬──────┘
               ▼               ▼
          as_completed() yields
          futures as they finish
               │
               ▼
        f.result() gets output
               │
               ▼
      evidence = {
          "probing": ...,
          "adoption": ...,
          "prediction": ...
      }

Run:
    ../1.KPI_Narrator/.venv/bin/python 2.parallel_agents.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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


def call_llm(system: str, user: str) -> str:
    return _text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)


probing_system = 'You are the Probing Agent. Flag KPI anomalies. Return JSON: {"signals": [...]}'
adoption_system = 'You are the Adoption Agent. Assess commitment risk. Return JSON: {"risk": "..."}'
prediction_system = 'You are the Prediction Agent. Forecast the trend. Return JSON: {"forecast": "..."}'
narrative_system = (
    "You are the Narrative Agent for the CS Lead. Use ONLY the evidence package. "
    "Cite metric IDs. Max 4 sentences."
)


def run_probing(data):
    return call_llm(probing_system, json.dumps(data))


def run_adoption(data):
    return call_llm(adoption_system, json.dumps(data))


def run_prediction(data):
    return call_llm(prediction_system, json.dumps(data))


def main() -> None:
    kpi_data = {"D-110": {"value": 12000, "prior": 24000}}

    # Fan out: submit all three at once; collect as each finishes.
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(run_probing, kpi_data): "probing",
            pool.submit(run_adoption, {"C-09": "amber"}): "adoption",
            pool.submit(run_prediction, kpi_data): "prediction",
        }
        evidence = {}
        for f in as_completed(futures):
            evidence[futures[f]] = f.result()
    print(f"gathered 3 agents in parallel in {time.time() - start:.1f}s\n")
    print("EVIDENCE PACKAGE:")
    print(json.dumps(evidence, indent=2), "\n")

    # Merge -> one narrative.
    final = call_llm(narrative_system, f"EVIDENCE PACKAGE:\n{json.dumps(evidence, indent=2)}")
    print("── Narrative Agent (final) ──\n", final)
    print(
        "\nAll three specialists ran concurrently, then a single Narrative agent\n"
        "merged their outputs. This is the 'parallel gather, then narrate' shape\n"
        "used for fast trigger-to-artifact orchestration."
    )


if __name__ == "__main__":
    main()


"""
Output:

(wise_albert_env) venkatesh@venkatesh:~/WiseAlbert/PracticeAI/6.Multi_Agent_Systems$ ../1.KPI_Narrator/.venv/bin/python 2.parallel_agents.py

gathered 3 agents in parallel in 59.5s

EVIDENCE PACKAGE:
{
  "adoption": "{\"risk\": \"low\"}",
  "prediction": "{\"forecast\": \"decreasing\"}",
  "probing": "```json\n{\"signals\": [{\"metric\": \"D-110\", \"anomaly_type\": \"value_drop\", \"anomaly_score\": 0.5}]}\n```"
} 

── Narrative Agent (final) ──
 The adoption risk is low. However, the forecast predicts a decreasing trend. Probing revealed a value drop in metric D-110 with an anomaly score of 0.5.

All three specialists ran concurrently, then a single Narrative agent
merged their outputs. This is the 'parallel gather, then narrate' shape
used for fast trigger-to-artifact orchestration.

"""