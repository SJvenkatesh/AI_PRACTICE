"""
Example H — Wise Albert Context Thread (Personal Intel agent)
=============================================================

The capstone memory pattern: a persistent "Context Thread" — a running log of
what was decided, promised, released, and discussed with a stakeholder, drawn
from many sources (emails, Jira, Confluence, past alerts, prior call decisions)
and updated after every interaction.

When the CS Lead opens a pre-call brief, the agent retrieves the recent thread
entries and turns them into a briefing. It's long-term + episodic memory
(Examples D & E) applied to a real workflow.

    Context Thread (persistent log)  →  recent entries  →  LLM  →  pre-call brief

Provider: free Gemini (flash-lite). One LLM call.


Instead of treating every conversation independently, maintain a persistent timeline of everything important about a customer or stakeholder.



Imagine you're a Customer Success (CS) Lead

You manage a customer called Acme Corp.

Over the last six months:

Emails were exchanged
Jira tickets were created
Bugs were fixed
Promises were made
Meetings happened
Alerts were generated

You don't want the AI to forget all of that.

So you maintain one running history.

Context Thread

Think of it as a diary.

Jan 10
Customer requested SDK support.

Jan 18
Engineering promised release in August.

Feb 2
Customer reported activation issue.

Mar 15
Jira-245 closed.

Apr 1
Customer escalated onboarding delays.

May 12
Release deployed.

Jun 20
Customer confirmed improvement.

This keeps growing.

Where does the data come from?

Many systems.

Emails
      │
      ▼
Jira
      │
      ▼
Confluence
      │
      ▼
Slack
      │
      ▼
Meeting Notes
      │
      ▼
Alerts
      │
      ▼
Context Thread

The Context Thread isn't typed manually.

The AI (or backend services) continuously updates it.

Example Entry

Instead of storing raw email:

Subject:
Meeting

Body:
Hi...
...
...

The system stores a concise memory:

2026-07-01

Customer requested SDK rollout by August.

Next meeting

2026-07-10

Engineering confirmed rollout is on schedule.

Next week

2026-07-18

Customer reported activation dropped 20%.

Over time:

2026-07-01
SDK promise

2026-07-10
Rollout confirmed

2026-07-18
Activation issue

2026-07-25
Hotfix released
Before a Meeting

Suppose tomorrow you have a customer call.

Instead of reading

300 emails
50 Jira tickets
20 Confluence pages

the AI does

Retrieve recent Context Thread

Maybe the last 20 entries.

Then it prompts the LLM

Here is the customer's context thread.

Summarize the important points.

Generate a pre-call briefing.

Output

Pre-call Brief

• SDK rollout promised for August.

• Customer remains concerned about activation.

• Jira-245 resolved.

• Follow up on onboarding metrics.
Why not store the whole email history?

Imagine

10,000 emails.

Sending all of them to GPT every time would be:

Slow
Expensive
Larger than the model's context window

Instead

Store

Important events

not

Everything
Long-term Memory

The document says

Long-term + episodic memory

Let's understand both.

Long-term Memory

Facts that remain true.

Example

Customer uses SDK v3.

Preferred communication:
Email.

Account owner:
Venkatesh.

These don't change often.

Episodic Memory

Specific events.

Example

2026-07-01

Customer requested rollout.
2026-07-12

Support escalation.
2026-07-20

Meeting action item.

These are events in time.

Context Thread = Timeline

Imagine

Customer Timeline

↓

Jan

↓

Feb

↓

Mar

↓

Apr

↓

May

Instead of

Random Notes

everything is chronological.

Retrieval

Before answering

the AI doesn't read everything.

It searches

Question

↓

Vector Search

↓

Recent Context Thread Entries

↓

LLM

For example

User asks

What commitments have we made?

The retriever finds

Jul 1

SDK rollout by August
Jul 8

Dashboard by September

Only those memories go to the LLM.

Real-world Analogy

Imagine your doctor.

Every visit is recorded.

2024

High blood pressure.

2025

Medication changed.

2025

Blood test normal.

2026

Follow-up recommended.

When you visit today,

the doctor doesn't reread every lab report.

They open your medical history and quickly review the timeline.

A Context Thread works the same way.

Architecture
          Emails
             │
             ▼
           Jira
             │
             ▼
        Confluence
             │
             ▼
      Meeting Notes
             │
             ▼
      Context Thread
      (Persistent Memory)
             │
      Retrieve Recent Entries
             │
             ▼
             LLM
             │
             ▼
       Pre-call Brief
Difference from a Vector Database

A Context Thread is not just a vector database.

A vector database answers:

"Which memories are semantically similar to this question?"

A Context Thread answers:

"What is the chronological story of this customer or stakeholder?"




Run:
    ../1.KPI_Narrator/.venv/bin/python 8.context_thread.py
"""

import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
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


# The persistent memory artifact — grows after every interaction with IOH.
context_thread = {
    "user_id": "cs_lead_1",
    "stakeholder": "IOH commercial",
    "entries": [
        {"date": "2026-07-01", "type": "commitment_received",
         "text": "IOH asked for SDK funnel by Aug", "id": "C-09"},
        {"date": "2026-07-05", "type": "decision",
         "text": "Escalated D-110 activation drop to Product"},
        {"date": "2026-07-07", "type": "release",
         "text": "Narrative Studio shipped internally"},
    ],
}


def build_precall_brief(thread: dict, question: str) -> str:
    recent = thread["entries"][-5:]   # only the recent thread entries
    resp = llm.invoke([HumanMessage(content=(
        f"You are prepping the CS Lead for a call with {thread['stakeholder']}.\n"
        f"CONTEXT THREAD (recent entries):\n{recent}\n\n"
        f"Prepare a short brief for: {question}\n"
        "Cite commitment IDs. Max 4 bullet points."
    ))])
    return _text(resp.content)


def main() -> None:
    question = "What should I raise in today's IOH call?"
    print(f"stakeholder: {context_thread['stakeholder']}")
    print("U:", question, "\n")
    print("PRE-CALL BRIEF:\n", build_precall_brief(context_thread, question))
    print(
        "\nThe brief was built from the persistent Context Thread — commitments,\n"
        "decisions, and releases logged over days — not from anything in this\n"
        "prompt session. That running log IS the memory."
    )


if __name__ == "__main__":
    main()
