"""
Example A — Tool schema (the "order slip")
==========================================

Before an LLM can call a tool, it needs a JSON SCHEMA describing it: the name,
what it does, and what arguments it takes. Think of it as an order slip the
model fills in — it never runs your code, it just produces a name + arguments
that match this schema; YOUR code runs the function.

The single most important field is `description`: say WHEN to use the tool, when
NOT to, and what it returns. That text is what the model reads to decide.

Two ways to get a schema:
  1. Hand-write the raw JSON (full control) — shown as HAND_WRITTEN below.
  2. Decorate a Python function with @tool and let LangChain generate it from the
     signature + docstring — shown via convert_to_openai_tool().

No LLM call — this just prints the schemas so you can see them.


Suppose you ask ChatGPT:

"What's the weather in Hyderabad?"

Can GPT answer this?

Not reliably.

Why?

Because the weather changes every minute.

Instead, GPT needs a weather API.

But GPT doesn't know:

What APIs exist
How to call them
What parameters they need

We must teach it.

That's exactly what a Tool Schema does.


Before an LLM can call a tool, it needs a JSON Schema describing it.

Suppose you have a Python function.

def get_weather(city):
    ...

GPT cannot magically understand this.

You have to describe it.

Like this:

{
  "name": "get_weather",

  "description": "Get current weather for a city.",

  "parameters": {
      "city": "string"
  }
}

This description is sent to GPT.


Run:
    ../1.KPI_Narrator/.venv/bin/python 1.tool_schema.py
"""

import json

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

# 1) Hand-written schema — exactly what gets sent to the model.
HAND_WRITTEN = {
    "type": "function",
    "function": {
        "name": "get_kpi",
        "description": (
            "Fetch current and prior-week value for a KPI metric. "
            "Use for live numbers like activation, ARPU, churn. "
            "Do NOT guess metrics — always call this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric_id": {"type": "string", "description": "Metric registry ID, e.g. D-110, D-150"},
                "weeks": {"type": "integer", "description": "How many weeks of history. Default 1."},
            },
            "required": ["metric_id"],
        },
    },
}


# 2) The same tool via @tool — the docstring becomes the description, and the
#    type hints become the parameters. LangChain generates the schema for you.
@tool
def get_kpi(metric_id: str, weeks: int = 1) -> dict:
    """Fetch current and prior-week value for a KPI metric. Use for live numbers
    like activation, ARPU, churn. Do NOT guess metrics — always call this.

    Args:
        metric_id: Metric registry ID, e.g. D-110, D-150.
        weeks: How many weeks of history. Default 1.
    """
    return {}  # body irrelevant here — we only care about the generated schema


def main() -> None:
    print("=== 1) HAND-WRITTEN schema ===")
    print(json.dumps(HAND_WRITTEN, indent=2))

    print("\n=== 2) @tool AUTO-GENERATED schema (from signature + docstring) ===")
    print(json.dumps(convert_to_openai_tool(get_kpi), indent=2))

    print(
        "\nBoth describe the same 'order slip'. The @tool version is generated from\n"
        "your function, so the code and the schema can't drift apart. Note how the\n"
        "docstring text carries the all-important WHEN/WHEN-NOT guidance."
    )


if __name__ == "__main__":
    main()

"""
=== 1) HAND-WRITTEN schema ===
{
  "type": "function",
  "function": {
    "name": "get_kpi",
    "description": "Fetch current and prior-week value for a KPI metric. Use for live numbers like activation, ARPU, churn. Do NOT guess metrics \u2014 always call this.",
    "parameters": {
      "type": "object",
      "properties": {
        "metric_id": {
          "type": "string",
          "description": "Metric registry ID, e.g. D-110, D-150"
        },
        "weeks": {
          "type": "integer",
          "description": "How many weeks of history. Default 1."
        }
      },
      "required": [
        "metric_id"
      ]
    }
  }
}

=== 2) @tool AUTO-GENERATED schema (from signature + docstring) ===
{
  "type": "function",
  "function": {
    "name": "get_kpi",
    "description": "Fetch current and prior-week value for a KPI metric. Use for live numbers\n    like activation, ARPU, churn. Do NOT guess metrics \u2014 always call this.\n\n    Args:\n        metric_id: Metric registry ID, e.g. D-110, D-150.\n        weeks: How many weeks of history. Default 1.",
    "parameters": {
      "properties": {
        "metric_id": {
          "type": "string"
        },
        "weeks": {
          "default": 1,
          "type": "integer"
        }
      },
      "required": [
        "metric_id"
      ],
      "type": "object"
    }
  }
}

Both describe the same 'order slip'. The @tool version is generated from
your function, so the code and the schema can't drift apart. Note how the
docstring text carries the all-important WHEN/WHEN-NOT guidance.
"""