"""
Example D — Metrics to track
============================

Logs tell you what happened in ONE request. Metrics aggregate across MANY:
latency (especially the p95 — the slow tail users actually feel), error counts,
token usage, and cost. You watch these on a dashboard and alert when they breach
(Example 7).

Targets to design against:
  - P95 query latency < 15s   (alert if exceeded)
  - Full pipeline    < 90s    (smoke test)
  - Track cost per weekly-digest run

No LLM — we feed the collector simulated numbers and print the summary.

Run:
    ../1.KPI_Narrator/.venv/bin/python 3.metrics.py
"""

import statistics
from collections import defaultdict


class MetricsCollector:
    def __init__(self):
        self.latencies = defaultdict(list)
        self.errors = defaultdict(int)
        self.token_counts = defaultdict(int)
        self.costs = []

    def record_latency(self, step: str, ms: float):
        self.latencies[step].append(ms)

    def record_error(self, step: str):
        self.errors[step] += 1

    def record_tokens(self, model: str, tokens_in: int, tokens_out: int):
        self.token_counts[f"{model}_in"] += tokens_in
        self.token_counts[f"{model}_out"] += tokens_out
        # Example rates (~$/1M tokens) — adjust to your model's real pricing.
        cost = (tokens_in * 0.15 + tokens_out * 0.60) / 1_000_000
        self.costs.append(cost)

    def p95_latency(self, step: str) -> float:
        data = self.latencies[step]
        # quantiles needs >=2 points; the n=20 trick gives the 95th percentile.
        if len(data) >= 20:
            return statistics.quantiles(data, n=20)[18]
        return max(data, default=0)

    def summary(self) -> dict:
        return {
            "p95_total_ms": round(self.p95_latency("total"), 1),
            "error_count": sum(self.errors.values()),
            "total_cost_usd": round(sum(self.costs), 4),
            "query_count": len(self.costs),
        }


def main() -> None:
    m = MetricsCollector()

    # Simulate 25 requests so p95 (needs >=20 samples) is meaningful.
    fake_totals = [4200, 5100, 3900, 6000, 4800, 5200, 4100, 7000, 4500, 5300,
                   3800, 4900, 5000, 6200, 4300, 4700, 5100, 8800, 4600, 5400,
                   4000, 5900, 4400, 5000, 16000]  # last one is a slow outlier
    for ms in fake_totals:
        m.record_latency("total", ms)
        m.record_tokens("gemini-flash-lite", tokens_in=1250, tokens_out=180)
    m.record_error("rag_search")   # one failure across the batch

    print("Summary:", m.summary())
    p95 = m.p95_latency("total")
    print(f"\nP95 total latency: {p95:.0f}ms", "⚠ over 15s SLA" if p95 > 15000 else "(within 15s SLA)")
    print(
        "\nThe p95 (not the average) surfaces the slow tail — one 16s outlier drags\n"
        "it up even though most requests were ~5s. That's the number you alert on."
    )


if __name__ == "__main__":
    main()
