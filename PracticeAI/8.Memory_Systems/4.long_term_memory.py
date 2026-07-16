"""
Example D — Long-term memory (persistent user facts)
====================================================

Short-term/window/summary memory all live within ONE session. Long-term memory
persists ACROSS sessions: durable facts about a user (role, focus, account) live
in a database and get injected into the system prompt at the start of any new
conversation — so the assistant "knows" the user without being re-told.

    DB row(s)  →  get_user_context()  →  system prompt  →  personalised answer

In production you'd extract these facts from conversations automatically (with
the user's consent) and write them back to the store.

Provider: free Gemini (flash-lite). One LLM call.

Run:
    ../1.KPI_Narrator/.venv/bin/python 4.long_term_memory.py
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


# Stand-in for a `user_memory` table. Persists between sessions in a real DB.
USER_MEMORY = {
    "venkatesh": [
        {"key": "role", "value": "software developer on Wise Albert engineering"},
        {"key": "focus", "value": "frontend + backend, learning AI"},
        {"key": "account", "value": "IOH / Wisely AI"},
    ]
}


def get_user_context(user_id: str) -> str:
    facts = USER_MEMORY.get(user_id, [])
    return "\n".join(f"- {f['key']}: {f['value']}" for f in facts)


def chat_with_long_term_memory(user_id: str, question: str) -> str:
    user_context = get_user_context(user_id)
    resp = llm.invoke([
        SystemMessage(content=f"You are Wise Albert.\n\nUSER PROFILE:\n{user_context}"),
        HumanMessage(content=question),
    ])
    return _text(resp.content)


def main() -> None:
    # The user never states who they are — the profile comes from long-term memory.
    q = "Given who I am, suggest one AI topic I should learn next and why."
    print("user_id: venkatesh")
    print("U:", q)
    print("A:", chat_with_long_term_memory("venkatesh", q))
    print(
        "\nThe answer is tailored to the stored profile (developer, learning AI,\n"
        "IOH account) even though the question never mentioned any of it — that's\n"
        "long-term memory injected at session start."
    )


if __name__ == "__main__":
    main()


"""
venkatesh@venkatesh:~/WiseAlbert/PracticeAI/8.Memory_Systems$ ../1.KPI_Narrator/.venv/bin/python 4.long_term_memory.py
user_id: venkatesh
U: Given who I am, suggest one AI topic I should learn next and why.
A: Given your role as a software developer at Wise Albert, with a focus on both frontend and backend, and your interest in learning AI, I recommend you dive into **"Fine-tuning Large Language Models (LLMs) for Specific Tasks."**

Here's why this is a perfect next step for you:

*   **Directly Applicable to Wise Albert:** As a company building AI-powered solutions, understanding how to tailor existing powerful LLMs to the specific needs of your products and users is incredibly valuable. This isn't just theoretical AI; it's about making AI work *better* for your specific context.
*   **Bridging Frontend and Backend:**
    *   **Backend:** You'll be dealing with the technical aspects of data preparation, model selection, training infrastructure (even if it's cloud-based), and deployment. This directly leverages your backend skills.
    *   **Frontend:** Understanding what a fine-tuned model can *do* and how to prompt it effectively will directly inform how you design user interfaces and experiences that leverage thesecapabilities. You'll be able to anticipate what's possible and build features around it.
*   **Leveraging Existing AI Advancements:** Instead of starting from scratch with model architecture, fine-tuning allows you to build upon the massive investments already made in foundational LLMs (like those from OpenAI, Google, Meta, etc.). This is a more practical and efficient way to get powerful AI capabilities into your hands.
*   **Learning Practical AI Engineering:** This topic involves a blend of theoretical understanding (how LLMs work at a high level, what fine-tuning actually changes) and practical implementation (data curation, hyperparameter tuning, evaluation metrics, deployment strategies). It's a great way to build tangible AI engineering skills.
*   **High Demand Skill:** The ability to effectively fine-tune LLMs is a highly sought-after skill in the current AI landscape. It allows companies to create specialized AI agents, improvechatbot responses, generate more relevant content, and much more, all without the immense costof training a model from scratch.
*   **Foundation for Future Learning:** Once you understand fine-tuning, you'll have a much better grasp of how to approach other advanced AI topics like prompt engineering, RAG (Retrieval Augmented Generation), model evaluation, and even the basics of model architecture.

**In essence, learning to fine-tune LLMs will equip you with the skills to directly contributeto Wise Albert's AI initiatives by making powerful AI models more relevant, efficient, and effective for your specific use cases.**

To get started, you could explore resources on:

*   **Parameter-Efficient Fine-Tuning (PEFT) techniques:** Like LoRA (Low-Rank Adaptation), which are more resource-friendly.
*   **Open-source LLM frameworks:** Such as Hugging Face Transformers, which provide excellenttools for fine-tuning.
*   **Data preparation strategies:** For supervised fine-tuning.
*   **Evaluation metrics:** To assess the performance of your fine-tuned models.

The answer is tailored to the stored profile (developer, learning AI,
IOH account) even though the question never mentioned any of it — that's
long-term memory injected at session start.

"""