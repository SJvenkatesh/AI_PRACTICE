"""
Example G — Persona resolver
============================

Same underlying data, DIFFERENT orchestration per persona. A CEO, a CS Lead, and
a Product manager asking the same question should get different agents run, a
different tone, a different template, and a different length.

The orchestrator reads a persona's config from a registry and uses it to drive
the agent list and the narrative step. Adding a new persona = a new registry
entry, not new code — the pipeline stays the same.

No LLM call — this shows the routing/config resolution only.

Run:
    ../1.KPI_Narrator/.venv/bin/python 7.persona_resolver.py
"""

PERSONA_CONFIG = {
    "ceo": {
        "tone": "executive, outcome-focused",
        "agents": ["probing", "adoption", "prediction"],
        "template": "exec_scorecard",
        "max_bullets": 5,
    },
    "cs_lead": {
        "tone": "operational, action-oriented",
        "agents": ["probing", "adoption", "personal_intel"],
        "template": "weekly_impact",
        "max_bullets": 8,
    },
    "product": {
        "tone": "technical, funnel-focused",
        "agents": ["adoption", "probing"],
        "template": "sdk_funnel_view",
        "max_bullets": 6,
    },
}


def resolve_persona(user_id: str) -> str:
    """In production: look up the persona from auth / session. Mocked here."""
    return {"user_123": "cs_lead", "user_ceo": "ceo", "user_pm": "product"}.get(user_id, "cs_lead")


def main() -> None:
    question = "How is the SDK funnel doing?"
    print("Same question, three personas:\n  ", question, "\n")

    for user_id in ("user_ceo", "user_123", "user_pm"):
        persona = resolve_persona(user_id)
        cfg = PERSONA_CONFIG[persona]
        print(f"── {persona} ({user_id}) ──")
        print(f"   agents  : {cfg['agents']}")
        print(f"   tone    : {cfg['tone']}")
        print(f"   template: {cfg['template']} (max {cfg['max_bullets']} bullets)")
        print()

    print(
        "Notice the agent SET differs per persona (product skips prediction;\n"
        "cs_lead adds personal_intel). The orchestrator feeds cfg['agents'] to the\n"
        "planner and cfg['template']/tone to the narrative step. New persona =\n"
        "one registry entry — modular by design."
    )


if __name__ == "__main__":
    main()
