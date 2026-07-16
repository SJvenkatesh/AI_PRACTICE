"""
Example A — Short-term memory: resend chat history
==================================================

The simplest form of memory: the LLM is stateless, so YOUR app keeps the
conversation and resends ALL prior messages on every call. The model "remembers"
only because you keep handing it the whole transcript.

    turn 1: [system, user1]                 → reply1
    turn 2: [system, user1, reply1, user2]  → reply2  (sees user1 → "remembers")

Store the history wherever you like (a list here; Redis/DB in production).

Problem: long chats blow past the context window and cost more every turn — which
is why Examples B (window) and C (summary) exist.

Provider: free Gemini (flash-lite).

Run:
    ../1.KPI_Narrator/.venv/bin/python 1.short_term_memory.py
"""

import os
import sys
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


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


# The whole memory lives in this list, in your app — not in the model.
conversation = [
    {"role": "system", "content": "You are the Wise Albert assistant for a CS Lead."},
]


def chat(user_message: str) -> str:
    conversation.append({"role": "user", "content": user_message})
    reply = _text(llm.invoke(conversation).content)   # resend the FULL history
    conversation.append({"role": "assistant", "content": reply})
    return reply


def main() -> None:
    print("U: I'm preparing for the IOH cadence call.")
    print("A:", chat("I'm preparing for the IOH cadence call."))

    print("\nU: What was I preparing for?")
    print("A:", chat("What was I preparing for?"))   # remembers — history was resent

    print(f"\n(history now holds {len(conversation)} messages — grows every turn,")
    print(" which is exactly the cost/context problem Examples B and C solve.)")


if __name__ == "__main__":
    main()


"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 1.short_term_memory.py
U: I'm preparing for the IOH cadence call.
A: Ah, the IOH cadence call! A crucial touchpoint for any CS Lead. It's where you align with your leadership, showcase your team's impact, and identify any roadblocks.

To help you prepare, let's break down what typically makes for a successful IOH (likely "Impact, Opportunities, and Health") cadence call. I'll assume this is a regular, recurring meeting.

**Before the Call: The Foundation of Your Preparation**

1.  **Understand the Goal of the IOH Call:** What is your leadership *really* looking for?
    *   **Impact:** How is your team contributing to the company's strategic objectives?
    *   **Opportunities:** Where can your team drive further growth, efficiency, or customer value?
    *   **Health:** What are the key indicators of your team's performance and customer satisfaction? Are there any risks or challenges?

2.  **Gather Your Data (This is Key!):** You need to be data-driven. What metrics are you tracking?
    *   **Customer Health Scores:** What's the overall trend? Are there specific segments that are improving or declining?
    *   **Churn/Retention Rates:** What are your numbers? Any significant wins or losses to highlight? What are the contributing factors?
    *   **Expansion/Upsell Revenue:** How is your team contributing to revenue growth?
    *   **Customer Satisfaction (CSAT/NPS):** What are the trends? Any specific feedback driving these scores?
    *   **Product Adoption/Engagement:** Are customers using the product effectively? Where are the adoption gaps?
    *   **Onboarding Success:** Are new customers getting value quickly?
    *   **Customer Feedback/Insights:** What are customers saying? What are the recurring themes?
    *   **Team Performance Metrics:** (If applicable) Onboarding time, resolution times, proactive outreach metrics, etc.

3.  **Synthesize and Analyze Your Data:** Don't just present raw numbers.
    *   **Identify Trends:** What's changing over time?
    *   **Draw Insights:** *Why* are these trends happening? What's the story behind the numbers?
    *   **Connect to Business Goals:** How do your team's metrics directly impact the company's OKRs or strategic priorities?

4.  **Prepare Your Narrative:** This is where you turn data into a compelling story.
    *   **Start with the Big Picture:** Briefly reiterate your team's mission and how it aligns with leadership's priorities.
    *   **Highlight Key Wins/Impact:** What are the most significant achievements your team has had since the last call? Quantify them!
    *   **Discuss Opportunities:** Where do you see potential for growth or improvement? What initiatives are you planning?
    *   **Address Challenges/Risks:** What are the obstacles you're facing? What support do you need? Be proactive in identifying these.
    *   **Customer Stories (Optional but Powerful):** A brief, impactful customer success story can humanize your data and demonstrate value.

5.  **Anticipate Questions:** Put yourself in your leadership's shoes. What will they ask?
    *   "What's driving the increase/decrease in X metric?"
    *   "What are the biggest risks to our customer base?"
    *   "How are you addressing the feedback about Y feature?"
    *   "What resources do you need to capitalize on this opportunity?"
    *   "What's the ROI of your team's initiatives?"

6.  **Define Your "Asks":** What do you need from leadership?
    *   Resources (headcount, budget, tools)
    *   Decision-making authority
    *   Cross-functional alignment or support
    *   Guidance on strategic priorities

**During the Call: Delivering Your Message**

1.  **Be Concise and Focused:** Respect everyone's time. Get to the point quickly.
2.  **Lead with Impact:** Start with your most significant achievements and their business impact.
3.  **Use Visuals Effectively:** Charts, graphs, and dashboards can make complex data easier to understand.
4.  **Be Prepared to Dive Deeper:** Have supporting data and details readily available if asked.
5.  **Listen Actively:** Pay attention to questions and feedback.
6.  **Be Transparent and Honest:** Don't shy away from challenges, but present them with proposed solutions.
7.  **Collaborate:** Frame discussions as opportunities for partnership and problem-solving.
8.  **Confirm Next Steps:** Ensure everyone is clear on actions, owners, and timelines.

**After the Call: Follow-Up and Continuous Improvement**

1.  **Distribute Minutes/Action Items:** Share a summary of the discussion and agreed-upon next steps.
2.  **Execute on Your Commitments:** Follow through on any actions you've committed to.
3.  **Reflect and Iterate:** What went well? What could be improved for the next call?

**To help me tailor this advice further, tell me:**

*   **What is the specific agenda or typical structure of your IOH calls?**
*   **Who are the key stakeholders attending the call?** (e.g., VP of Sales, Head of Product, CEO)
*   **What are the top 1-2 strategic priorities for your company right now?**
*   **What are your biggest concerns or areas you want to highlight?**

Let's make sure you walk into that call feeling confident and prepared!

U: What was I preparing for?
A: You are preparing for your **IOH cadence call**.

Based on our previous conversation, this is a regular, recurring meeting where you, as a CS Lead, will be presenting to your leadership. The "IOH" likely stands for **Impact, Opportunities, and Health**, and the call's purpose is to:

*   **Showcase the impact** your Customer Success team is having on the business.
*   Discuss **opportunities** for growth, improvement, or further value creation.
*   Report on the overall **health** of your customer base and your team's performance.

Essentially, it's your opportunity to demonstrate the value of your team, highlight successes, identify challenges, and align with leadership on strategic direction.

(history now holds 5 messages — grows every turn,
 which is exactly the cost/context problem Examples B and C solve.)


"""