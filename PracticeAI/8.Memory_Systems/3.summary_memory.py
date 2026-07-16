"""
Example C — Summary memory (compress old history)
=================================================

Buffer window (Example B) forgets old facts entirely. Summary memory keeps them
in COMPRESSED form: when the history gets long, summarise the old turns into a
few bullet points, then send [system + summary + last few raw messages]. You
keep the gist of the whole conversation without resending 50 pages.

    [ system ] + [ SUMMARY of old turns ] + [ last N raw messages ] + [ new Q ]

This is the pattern behind a running "context thread" of decisions.

Provider: free Gemini (flash-lite). Two LLM calls (summarise + answer).

Run:
    ../1.KPI_Narrator/.venv/bin/python 3.summary_memory.py
"""

import os
import sys
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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def summarize_history(old_messages: list) -> str:
    """Compress old turns to 3 bullets, preserving facts and metric IDs."""
    resp = llm.invoke([
        SystemMessage(content=(
            "Summarize this conversation in 3 bullet points. "
            "Keep facts and metric/commitment IDs."
        )),
        HumanMessage(content=str(old_messages)),
    ])
    return _text(resp.content)


def main() -> None:
    # A long-ish history we don't want to resend verbatim every turn.
    old_messages = [
        {"role": "user", "content": "Activation KPI D-110 fell from 24k to 12k."},
        {"role": "assistant", "content": "Noted — a 50% week-over-week drop on D-110."},
        {"role": "user", "content": "Commitment C-09 (SDK funnel by Aug) is now amber."},
        {"role": "assistant", "content": "Recorded C-09 as amber."},
        {"role": "user", "content": "We escalated the D-110 drop to the Product team."},
        {"role": "assistant", "content": "Logged the escalation to Product."},
    ]

    summary = summarize_history(old_messages)
    print("SUMMARY of old turns:\n", summary, "\n")

    # New question — sent with the summary instead of the full old history.
    new_question = "Given the above, what should I flag in today's IOH call?"
    messages = [
        SystemMessage(content="You are Wise Albert. Cite metric/commitment IDs."),
        SystemMessage(content=f"CONVERSATION SUMMARY:\n{summary}"),
        # (in a real loop you'd also append the last 2-4 raw messages here)
        HumanMessage(content=new_question),
    ]
    print("U:", new_question)
    print("A:", _text(llm.invoke(messages).content))
    print(
        "\nThe answer used the compressed summary — the model still knows about\n"
        "D-110 and C-09 without us resending the entire transcript."
    )


if __name__ == "__main__":
    main()
