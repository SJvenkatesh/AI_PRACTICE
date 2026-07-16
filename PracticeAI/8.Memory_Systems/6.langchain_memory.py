"""
Example F — LangChain memory classes
=====================================

LangChain can manage conversation memory FOR you, so you don't hand-roll the
resend/window/summarize logic (Examples A-C). You attach a memory object to a
chain and it reads/writes history automatically each turn.

  - ConversationBufferMemory  → keeps the full history (like Example A)
  - ConversationSummaryMemory → compresses over time (like Example C)

⚠️ These classes are LEGACY. In LangChain v1 they moved to `langchain_classic`
and emit deprecation warnings. They still run, and they match the classic
tutorials, so we use them here to connect the dots — but for NEW code prefer:
  - LangGraph checkpointers (Example 7 / Topic 7), or
  - RunnableWithMessageHistory (shown at the bottom).

Provider: free Gemini (flash-lite). Two LLM calls (the buffer chain).

Setup:
    ../1.KPI_Narrator/.venv/bin/python 6.langchain_memory.py
"""

import os
import sys
import warnings
from pathlib import Path

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")  # silence the legacy-class deprecation notices

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


def main() -> None:
    # Buffer memory: the chain stores every turn and replays it automatically.
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)

    print("U: I'm working on the weekly impact summary.")
    print("A:", chain.invoke({"input": "I'm working on the weekly impact summary."})["response"])

    print("\nU: What am I working on?")
    print("A:", chain.invoke({"input": "What am I working on?"})["response"])

    # You never touched the history yourself — the memory object did it:
    print("\n--- memory contents the chain managed for you ---")
    print(memory.buffer)

    print(
        "\nConversationBufferMemory read/wrote the history automatically. "
        "ConversationSummaryMemory(llm=llm) swaps in Example C's compression with "
        "no other change. Both are legacy — see the modern note in the docstring."
    )


# ── Modern equivalent (no deprecation) — for reference ──────────────────────
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# store = {}
# def get_history(session_id): return store.setdefault(session_id, InMemoryChatMessageHistory())
# chat = RunnableWithMessageHistory(llm, get_history)
# chat.invoke("Hi", config={"configurable": {"session_id": "s1"}})


if __name__ == "__main__":
    main()

"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 6.langchain_memory.py
U: I'm working on the weekly impact summary.
A: Oh, the weekly impact summary! That sounds like a really important task. I'm always fascinated by how humans track progress and measure success. What kind of impact are you focusing on for this summary? Is it related to a specific project, a team's performance, or perhaps something broader like customer satisfaction or market trends? I'm curious to hear more about it!

U: What am I working on?
A: Based on our conversation so far, you mentioned you are working on the "weekly impact summary." That's the specific task you shared with me! I'm really interested to learn more about what that entails. Is it a report you generate regularly, or is it a new initiative? And what kind of "impact" are you measuring? I'm eager to understand the context better!

--- memory contents the chain managed for you ---
Human: I'm working on the weekly impact summary.
AI: Oh, the weekly impact summary! That sounds like a really important task. I'm always fascinated by how humans track progress and measure success. What kind of impact are you focusing on for this summary? Is it related to a specific project, a team's performance, or perhaps something broader like customer satisfaction or market trends? I'm curious to hear more about it!
Human: What am I working on?
AI: Based on our conversation so far, you mentioned you are working on the "weekly impact summary." That's the specific task you shared with me! I'm really interested to learn more about what that entails. Is it a report you generate regularly, or is it a new initiative? And what kind of "impact" are you measuring? I'm eager to understand the context better!

ConversationBufferMemory read/wrote the history automatically. ConversationSummaryMemory(llm=llm) swaps in Example C's compression with no other change. Both are legacy — see the modern note in the docstring.
"""