"""
Example B — Buffer window memory (last K messages)
==================================================

Short-term memory (Example A) grows forever. The simplest cap: keep only the
last K turns and let older ones fall off. A `deque(maxlen=...)` does this
automatically — appending past the cap drops the oldest item.

    MAX_TURNS = 2  →  the model only ever sees the 2 most recent turns.

Tradeoff: old context is dropped SILENTLY. Fine for short sessions or when only
recent context matters; use summary memory (Example C) when older facts still
need to survive.

Provider: free Gemini (flash-lite).

Run:
    ../1.KPI_Narrator/.venv/bin/python 2.buffer_window.py
"""

import os
import sys
from collections import deque
from pathlib import Path

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

MAX_TURNS = 2  # keep only the last 2 user+assistant pairs
history = deque(maxlen=MAX_TURNS * 2)  # 2 messages (user+assistant) per turn


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def chat_with_window(user_message: str) -> str:
    system = {"role": "system", "content": "You are Wise Albert. Answer briefly."}
    history.append({"role": "user", "content": user_message})
    messages = [system] + list(history)   # system is always kept; window is the rest
    reply = _text(llm.invoke(messages).content)
    history.append({"role": "assistant", "content": reply})
    return reply


def main() -> None:
    # Turn 1 states a fact that will later fall OUT of the 2-turn window.
    print("U: My name is Venkatesh and I work on IOH.")
    print("A:", chat_with_window("My name is Venkatesh and I work on IOH."))
    print("\nU: The SDK funnel commitment is C-09.")
    print("A:", chat_with_window("The SDK funnel commitment is C-09."))
    print("\nU: Activation dropped 50% last week.")
    print("A:", chat_with_window("Activation dropped 50% last week."))

    # By now turn 1 has been dropped from the window, so the model no longer
    # knows the name — this demonstrates the silent-forgetting tradeoff.
    print("\nU: What is my name?")
    print("A:", chat_with_window("What is my name?"))
    print(
        "\nThe window only held the last 2 turns, so 'Venkatesh' fell off and the\n"
        "model can't recall it. That silent drop is the cost of buffer-window memory."
    )


if __name__ == "__main__":
    main()
