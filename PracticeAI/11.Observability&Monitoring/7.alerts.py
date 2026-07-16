"""
Example H — Alerts (production monitoring)
==========================================

Metrics (Example 3) are only useful if something WATCHES them and shouts when a
threshold breaks. Alerts compare the aggregated metrics against SLA/OKR limits
and produce messages you'd route to Slack / PagerDuty.

No LLM. Run:
    ../1.KPI_Narrator/.venv/bin/python 7.alerts.py
"""

ALERT_THRESHOLDS = {
    "p95_latency_ms": 15_000,       # latency SLA
    "error_rate_pct": 5.0,
    "citation_rate_min": 0.95,      # quality OKR: 95% of answers cite sources
    "cost_per_query_usd": 0.05,
    "retrieval_zero_results": 3,    # 3 consecutive zero-result queries
}


def check_alerts(metrics: dict) -> list:
    alerts = []
    if metrics.get("p95_latency_ms", 0) > ALERT_THRESHOLDS["p95_latency_ms"]:
        alerts.append(f"⚠️ P95 latency {metrics['p95_latency_ms']}ms exceeds 15s SLA")
    if metrics.get("citation_rate", 1.0) < ALERT_THRESHOLDS["citation_rate_min"]:
        alerts.append(f"⚠️ Citation rate {metrics['citation_rate']:.0%} below 95% OKR")
    if metrics.get("error_rate_pct", 0) > ALERT_THRESHOLDS["error_rate_pct"]:
        alerts.append(f"⚠️ Error rate {metrics['error_rate_pct']}% too high")
    if metrics.get("cost_per_query_usd", 0) > ALERT_THRESHOLDS["cost_per_query_usd"]:
        alerts.append(f"⚠️ Cost/query ${metrics['cost_per_query_usd']:.3f} over budget")
    if metrics.get("retrieval_zero_results", 0) >= ALERT_THRESHOLDS["retrieval_zero_results"]:
        alerts.append(f"⚠️ {metrics['retrieval_zero_results']} consecutive zero-result retrievals")
    return alerts


def notify_slack(alerts: list) -> None:
    # Stand-in — real code would POST to a Slack webhook.
    for a in alerts:
        print(f"  [slack #alerts] {a}")


def main() -> None:
    healthy = {"p95_latency_ms": 8200, "citation_rate": 0.98, "error_rate_pct": 1.2,
               "cost_per_query_usd": 0.012, "retrieval_zero_results": 0}
    breaching = {"p95_latency_ms": 16800, "citation_rate": 0.88, "error_rate_pct": 7.5,
                 "cost_per_query_usd": 0.061, "retrieval_zero_results": 3}

    print("Healthy metrics → alerts:", check_alerts(healthy) or "none ✅")

    print("\nBreaching metrics → alerts:")
    alerts = check_alerts(breaching)
    notify_slack(alerts)

    print(
        "\nEach breached threshold became one actionable Slack line. In production\n"
        "you'd run check_alerts() on a schedule (or per request) and route the\n"
        "list to your alert channel — the SLAs/OKRs live in ALERT_THRESHOLDS."
    )


if __name__ == "__main__":
    main()
