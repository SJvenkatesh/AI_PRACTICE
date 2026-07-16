"""
Example E — Tool catalog (by agent)
===================================

As a system grows you don't give every agent every tool. Each agent gets a small,
focused catalog (rule of thumb: 3-8 tools). Some tools are READ-only and safe to
auto-run; others are WRITE tools (change state) or GUARDED (external / gated
behind an agreement) and need a permission check first.

This file is a catalog registry + a couple of sanity checks (the 3-8 rule, and
flagging tools that need auth). No LLM — pure structure.

Run:
    ../1.KPI_Narrator/.venv/bin/python 5.tool_catalog.py
"""

# Each tool entry: name + access class. "read" = safe; "write"/"guarded" = gated.
TOOL_CATALOG = {
    "probing": [
        {"name": "get_kpi", "access": "read"},
        {"name": "get_fp_fn_rates", "access": "read"},
        {"name": "rag_search", "access": "read"},
        {"name": "web_search", "access": "guarded"},          # external feeds
    ],
    "adoption": [
        {"name": "get_sdk_funnel", "access": "read"},
        {"name": "get_commitment", "access": "read"},
        {"name": "update_commitment_status", "access": "write"},  # needs auth
    ],
    "prediction": [
        {"name": "get_kpi_timeseries", "access": "read"},
        {"name": "forecast", "access": "read"},
        {"name": "detect_anomaly", "access": "read"},
    ],
    "personal_intel": [
        {"name": "search_outlook", "access": "guarded"},      # blocked pre-agreement
        {"name": "search_jira", "access": "read"},
        {"name": "search_confluence", "access": "read"},
    ],
}


def audit_catalog(catalog: dict) -> None:
    for agent, tools in catalog.items():
        names = [t["name"] for t in tools]
        gated = [t["name"] for t in tools if t["access"] in ("write", "guarded")]
        size_ok = "ok" if 3 <= len(tools) <= 8 else "⚠ outside 3-8"
        print(f"{agent:15} {len(names)} tools [{size_ok}]")
        print(f"                tools : {', '.join(names)}")
        if gated:
            print(f"                gated : {', '.join(gated)}  (permission check required)")
        print()


def tools_for(agent: str) -> list:
    """What the orchestrator would bind for a given agent (names only)."""
    return [t["name"] for t in TOOL_CATALOG.get(agent, [])]


def main() -> None:
    print("=== Tool catalog audit ===\n")
    audit_catalog(TOOL_CATALOG)
    print("Example — orchestrator binds only the adoption agent's tools:")
    print("  ", tools_for("adoption"))
    print(
        "\nKeeping each agent to a focused 3-8 tool set improves tool selection\n"
        "(the model isn't overwhelmed), and marking write/guarded tools tells the\n"
        "orchestrator which calls must pass an auth/HITL check before running."
    )


if __name__ == "__main__":
    main()


"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/9.Tool_Calling$ ../1.KPI_Narrator/.venv/bin/python 5.tool_catalog.py 
=== Tool catalog audit ===

probing         4 tools [ok]
                tools : get_kpi, get_fp_fn_rates, rag_search, web_search
                gated : web_search  (permission check required)

adoption        3 tools [ok]
                tools : get_sdk_funnel, get_commitment, update_commitment_status
                gated : update_commitment_status  (permission check required)

prediction      3 tools [ok]
                tools : get_kpi_timeseries, forecast, detect_anomaly

personal_intel  3 tools [ok]
                tools : search_outlook, search_jira, search_confluence
                gated : search_outlook  (permission check required)

Example — orchestrator binds only the adoption agent's tools:
   ['get_sdk_funnel', 'get_commitment', 'update_commitment_status']

Keeping each agent to a focused 3-8 tool set improves tool selection
(the model isn't overwhelmed), and marking write/guarded tools tells the
orchestrator which calls must pass an auth/HITL check before running.
"""