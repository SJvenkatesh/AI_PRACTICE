"""
Example F — Parameter validation (your code, not the LLM)
=========================================================

The model's tool arguments are UNTRUSTED input — it can hallucinate a malformed
metric_id or an out-of-range number, and a prompt-injected doc could steer it.
So validate every argument in YOUR code before you act on it, and never
string-concatenate LLM output into SQL/shell — use parameterized queries.

Here pydantic enforces the shape: metric_id must match `D-###`, weeks must be
1-52. Bad args return a clean error the model can read and recover from, instead
of hitting your database.

No LLM — we call the validated tool with good and bad arguments to show both.

Run:
    ../1.KPI_Narrator/.venv/bin/python 6.parameter_validation.py
"""

from pydantic import BaseModel, Field, ValidationError


class GetKpiArgs(BaseModel):
    metric_id: str = Field(pattern=r"^D-\d{3}$")   # only the D-### format
    weeks: int = Field(default=1, ge=1, le=52)     # 1..52 inclusive


def query_kpi_from_db(metric_id: str, weeks: int) -> dict:
    """Stand-in for a PARAMETERIZED DB query. In real code:
        cursor.execute("SELECT ... WHERE metric_id = %s AND weeks = %s",
                        (metric_id, weeks))
    Never f-string the LLM's value into the SQL text."""
    demo = {"D-110": {"name": "SDK activations", "value": 12000, "prior": 24000}}
    return {"metric_id": metric_id, "weeks": weeks, **demo.get(metric_id, {"value": None})}


def safe_get_kpi(metric_id: str, weeks: int = 1) -> dict:
    """Validate first; only touch the DB if the args are well-formed."""
    try:
        args = GetKpiArgs(metric_id=metric_id, weeks=weeks)
    except ValidationError as e:
        # Return a short, model-readable error — not a stack trace.
        problems = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        return {"error": f"Invalid arguments: {problems}"}
    return query_kpi_from_db(args.metric_id, args.weeks)


def main() -> None:
    cases = [
        {"metric_id": "D-110", "weeks": 1},      # valid
        {"metric_id": "D-110", "weeks": 999},    # weeks out of range
        {"metric_id": "DROP TABLE kpi;--"},      # not the D-### format (injection-y)
        {"metric_id": "D-999", "weeks": 4},      # valid shape, just no data
    ]
    for c in cases:
        print(f"args={c}")
        print(f"  -> {safe_get_kpi(**c)}\n")

    print(
        "The malformed and out-of-range calls were rejected BEFORE reaching the DB,\n"
        "returning a clean error the model can explain. The last one passed\n"
        "validation (correct shape) and simply had no data — a different case from\n"
        "a bad argument."
    )


if __name__ == "__main__":
    main()


"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/9.Tool_Calling$ ../1.KPI_Narrator/.venv/bin/python 6.parameter_validation.py
args={'metric_id': 'D-110', 'weeks': 1}
  -> {'metric_id': 'D-110', 'weeks': 1, 'name': 'SDK activations', 'value': 12000, 'prior': 24000}

args={'metric_id': 'D-110', 'weeks': 999}
  -> {'error': 'Invalid arguments: weeks: Input should be less than or equal to 52'}

args={'metric_id': 'DROP TABLE kpi;--'}
  -> {'error': "Invalid arguments: metric_id: String should match pattern '^D-\\d{3}$'"}

args={'metric_id': 'D-999', 'weeks': 4}
  -> {'metric_id': 'D-999', 'weeks': 4, 'value': None}

The malformed and out-of-range calls were rejected BEFORE reaching the DB,
returning a clean error the model can explain. The last one passed
validation (correct shape) and simply had no data — a different case from
a bad argument.

"""