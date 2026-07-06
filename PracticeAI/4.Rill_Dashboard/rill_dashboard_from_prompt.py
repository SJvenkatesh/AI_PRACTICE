"""
Rill Dashboard from a Prompt — Mini Project, Topic 4
====================================================

Question: "Based on a prompt, can we create a dashboard in Rill?"
Answer:   Yes. Rill is *BI-as-code* — a dashboard is nothing but two YAML files
          in a Rill project directory, which Rill hot-reloads on save:

            1. a **metrics view**  (the semantic layer: dimensions + measures)
            2. an **explore dashboard** (what the user actually interacts with)

So "prompt -> dashboard" = use an LLM to turn a plain-English request + the
table's real columns into those two YAML files, then write them into the
project. Rill picks them up automatically.

The production discipline (same as KPI_Narrator, Topic 1): the LLM only decides
*structure* — which columns become dimensions, which aggregations become
measures. It must use ONLY the columns we hand it and must NOT invent names or
data. We validate its output against a strict schema, then WE serialize the YAML
deterministically. The model phrases; the code is the system of record.

Setup
-----
    pip install langchain langchain-google-genai pydantic pyyaml python-dotenv
    cp ../1.KPI_Narrator/.env .   # reuse your GOOGLE_API_KEY
    python rill_dashboard_from_prompt.py

Then in Rill:  rill start   (from RILL_PROJECT_DIR) and open the new dashboard.

Docs: https://docs.rilldata.com/reference/project-files/metrics-views
      https://docs.rilldata.com/reference/project-files/explore-dashboards
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL = "gemini-2.5-flash"

# Where your Rill project lives. Metrics views and dashboards are just files
# under this directory; `rill start` watches them and reloads on change.
RILL_PROJECT_DIR = os.getenv("RILL_PROJECT_DIR", "./my-rill-project")


# --- The schema the LLM must fill (this is the guardrail) --------------------
# Rill's real YAML supports many more fields; these are the essential ones. By
# forcing the model into this shape we get validation for free and can serialize
# clean YAML ourselves instead of trusting the model to format it.
class Dimension(BaseModel):
    name: str = Field(description="stable snake_case identifier")
    column: str = Field(description="EXISTING column name to slice by")
    display_name: str
    description: str = ""


class Measure(BaseModel):
    name: str = Field(description="stable snake_case identifier")
    expression: str = Field(
        description="SQL aggregation over EXISTING columns, e.g. SUM(amount), COUNT(*)"
    )
    display_name: str
    format_preset: str = Field(
        default="humanize",
        description="one of: humanize, none, currency_usd, percentage, comma",
    )
    description: str = ""


class DashboardSpec(BaseModel):
    """Everything needed to emit both YAML files."""

    title: str = Field(description="human-friendly dashboard title")
    timeseries: str = Field(description="the timestamp column to use as time axis")
    dimensions: list[Dimension]
    measures: list[Measure]


SYSTEM = (
    "You are a Rill BI dashboard designer. Given a business request and the "
    "EXACT columns of a data model, choose sensible dimensions (categorical/time "
    "columns to slice by) and measures (SQL aggregations that answer the "
    "request). Rules: use ONLY the column names provided — never invent columns. "
    "Every measure expression must reference only those columns (COUNT(*) is "
    "allowed). Pick one timestamp column for `timeseries`. Prefer 3-6 dimensions "
    "and 2-5 measures. Use snake_case for every `name`."
)


def _slug(text: str) -> str:
    """Filesystem/Rill-safe resource name derived from the title."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "dashboard"


def generate_spec(prompt: str, model_name: str, columns: list[str]) -> DashboardSpec:
    """Ask Gemini to turn (prompt + real columns) into a validated DashboardSpec.

    Structured output means the model is forced to return our schema; LangChain
    validates it, so we never hand-parse free text.
    """
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    structured = llm.with_structured_output(DashboardSpec)
    user = (
        f"Business request:\n{prompt}\n\n"
        f"Data model name: {model_name}\n"
        f"Available columns: {', '.join(columns)}\n\n"
        "Design the dashboard."
    )
    return structured.invoke([("system", SYSTEM), ("human", user)])


def _metrics_view_yaml(spec: DashboardSpec, model_name: str) -> str:
    doc = {
        "version": "1",
        "type": "metrics_view",
        "display_name": spec.title,
        "model": model_name,
        "timeseries": spec.timeseries,
        "dimensions": [
            {
                "name": d.name,
                "column": d.column,
                "display_name": d.display_name,
                **({"description": d.description} if d.description else {}),
            }
            for d in spec.dimensions
        ],
        "measures": [
            {
                "name": m.name,
                "expression": m.expression,
                "display_name": m.display_name,
                "format_preset": m.format_preset,
                **({"description": m.description} if m.description else {}),
            }
            for m in spec.measures
        ],
    }
    return yaml.safe_dump(doc, sort_keys=False, width=100)


def _explore_yaml(spec: DashboardSpec, metrics_view_name: str) -> str:
    doc = {
        "type": "explore",
        "display_name": spec.title,
        "metrics_view": metrics_view_name,
        "dimensions": ["*"],  # expose all; or [d.name for d in spec.dimensions]
        "measures": ["*"],
    }
    return yaml.safe_dump(doc, sort_keys=False, width=100)


def create_rill_dashboard(
    prompt: str,
    model_name: str,
    columns: list[str],
    project_dir: str = RILL_PROJECT_DIR,
) -> dict:
    """Create a Rill dashboard from a natural-language prompt.

    Args:
        prompt:     what the dashboard should show, in plain English.
        model_name: name of an EXISTING Rill model/source (the .sql/source file
                    whose output these files will sit on top of).
        columns:    the real column names of that model (the LLM may use only
                    these — this is what keeps it from hallucinating fields).
        project_dir: root of the Rill project; files are written under it.

    Returns:
        dict with the generated spec and the paths of the two files written.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit(
            "GOOGLE_API_KEY is not set. Reuse the key from 1.KPI_Narrator/.env"
        )

    spec = generate_spec(prompt, model_name, columns)

    slug = _slug(spec.title)
    metrics_view_name = f"{slug}_metrics"

    root = Path(project_dir)
    mv_path = root / "metrics" / f"{metrics_view_name}.yaml"
    dash_path = root / "dashboards" / f"{slug}.yaml"
    mv_path.parent.mkdir(parents=True, exist_ok=True)
    dash_path.parent.mkdir(parents=True, exist_ok=True)

    mv_path.write_text(_metrics_view_yaml(spec, model_name))
    dash_path.write_text(_explore_yaml(spec, metrics_view_name))

    return {
        "spec": spec,
        "metrics_view_file": str(mv_path),
        "dashboard_file": str(dash_path),
    }


# --- Sample prompt + a demo run ---------------------------------------------
# A realistic request against an imaginary `orders` model. Swap the columns for
# your real model's columns when you wire this to a live Rill project.
SAMPLE_PROMPT = (
    "Build a sales performance dashboard for the leadership team. I want to track "
    "total revenue, number of orders, and average order value over time, and be "
    "able to break everything down by product category, sales region, and "
    "customer segment."
)

SAMPLE_MODEL = "orders"
SAMPLE_COLUMNS = [
    "order_id",
    "order_ts",
    "amount",
    "product_category",
    "sales_region",
    "customer_segment",
    "payment_method",
]


def main() -> None:
    print("Prompt:\n  " + SAMPLE_PROMPT + "\n")
    result = create_rill_dashboard(SAMPLE_PROMPT, SAMPLE_MODEL, SAMPLE_COLUMNS)

    spec: DashboardSpec = result["spec"]
    print(f"Title:      {spec.title}")
    print(f"Time axis:  {spec.timeseries}")
    print(f"Dimensions: {', '.join(d.name for d in spec.dimensions)}")
    print(f"Measures:   {', '.join(m.name for m in spec.measures)}\n")

    print("Wrote:")
    print(f"  metrics view -> {result['metrics_view_file']}")
    print(f"  dashboard    -> {result['dashboard_file']}\n")

    print("--- metrics view YAML ---")
    print(_metrics_view_yaml(spec, SAMPLE_MODEL))
    print("--- explore dashboard YAML ---")
    print(_explore_yaml(spec, f"{_slug(spec.title)}_metrics"))

    print("Next: run `rill start` from the project dir to see it live.")


if __name__ == "__main__":
    main()
