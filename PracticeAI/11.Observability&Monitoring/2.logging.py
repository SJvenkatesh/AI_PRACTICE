"""
Example C — Logging best practices
==================================

Every log line for one request should carry its trace_id, so you can grep all
the steps of a single user request out of a busy log. A LoggerAdapter injects
the trace_id automatically so you don't repeat it on every call.

Log levels — use them deliberately:
  INFO    normal steps (intent classified, agent dispatched, tool called)
  WARNING slow tool, low retrieval score, missing citation
  ERROR   tool failure, LLM rate limit, parse error

NEVER log API keys, PII, or raw billing data — redact before logging.

No LLM — pure logging. Run:
    ../1.KPI_Narrator/.venv/bin/python 2.logging.py
"""

import logging
import uuid

# NOTE: the format string does NOT reference %(trace_id)s. If you put a custom
# field in the format, every record must supply it or logging raises KeyError.
# The adapter below prefixes the trace_id into the message instead, which always
# works. (Alternative: pass extra={"trace_id": ...} and keep it in the format.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class TraceAdapter(logging.LoggerAdapter):
    """Prefixes every message with a short trace_id."""
    def process(self, msg, kwargs):
        return f"[trace={self.extra['trace_id'][:8]}] {msg}", kwargs


def redact(text: str) -> str:
    """Toy redactor — strip anything that looks like a key before logging."""
    import re
    return re.sub(r"(sk-[A-Za-z0-9_\-]{6,})", "sk-***REDACTED***", text)


def main() -> None:
    trace_id = str(uuid.uuid4())
    logger = TraceAdapter(logging.getLogger("wise_albert"), {"trace_id": trace_id})

    logger.info("Intent classified: kpi_anomaly")
    logger.info("Dispatching agents: probing, adoption")
    logger.warning("Tool get_kpi slow: 1200ms")
    logger.error("RAG returned 0 documents for query")

    # Redaction demo — never let a secret reach the logs.
    logger.info("Config loaded: " + redact("api_key=sk-abc123secretvalue used"))

    print(f"\n(every line above shares trace={trace_id[:8]} — grep that to see the")
    print(" whole request. The api_key was redacted before logging.)")


if __name__ == "__main__":
    main()
