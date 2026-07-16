"""
Example B — Full tool-calling loop
==================================

The complete cycle: give the model tools, it requests one (or more), YOUR code
runs it and returns the result, and the loop repeats until the model stops
asking and writes a final answer.

    system+user → model → (tool_calls?) → run tools → feed results → model → ...
                                                                          → answer

This is the same loop the agent topic used, viewed through the tool-calling
lens: the model only ever emits a tool NAME + ARGS; your `TOOL_FUNCTIONS` map
does the actual work.

Reference used OpenAI; this uses free Gemini via LangChain bind_tools.

Run:
    ../1.KPI_Narrator/.venv/bin/python 2.tool_calling_loop.py
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
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


@tool
def get_kpi(metric_id: str, weeks: int = 1) -> dict:
    #The docstring (the triple-quoted """...""" comment immediately below the function. LangChain inspects this function using Python's introspection.
    """Get live KPI values by metric ID (D-110, D-150). Never invent numbers."""
    db = {
        "D-110": {"name": "SDK activations", "value": 12000, "prior": 24000, "unit": "count"},
        "D-150": {"name": "ARPU", "value": 44000, "prior": 42000, "unit": "IDR"},
    }
    if metric_id not in db:
        return {"error": f"Metric {metric_id} not found"}
    return {"metric_id": metric_id, **db[metric_id], "weeks": weeks}


@tool
def rag_search(query: str, top_k: int = 3) -> list:
    """Search QBR notes and ops documents for context and commitments."""
    docs = [
        {"text": "D-110 SDK activation dropped after rollout paused in 2 regions", "source_id": "ops_w25"},
        {"text": "C-09 SDK funnel tracking status amber", "source_id": "qbr_q2"},
    ]
    words = query.lower().split()
    return [d for d in docs if any(w in d["text"].lower() for w in words)][:top_k]


TOOLS = [get_kpi, rag_search]
TOOL_FUNCTIONS = {t.name: t for t in TOOLS}
# Use flash (not flash-lite) for the HAND-ROLLED loop: after tool results,
# flash-lite sometimes ends the turn with empty text (finish_reason STOP, no
# answer). flash reliably writes the final answer. (Example 4's create_react_agent
# nudges the model, so flash-lite is fine there.)


#No, you do NOT need to call convert_to_openai_tool() yourself. bind_tools() does that internally.

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2).bind_tools(TOOLS)


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def run_tool_loop(user_question: str, max_steps: int = 5) -> str:
    messages = [
        SystemMessage(content=(
            "You are Wise Albert. Use tools for all facts. "
            "Cite metric_id and source_id. Never invent numbers."
        )),
        HumanMessage(content=user_question),
    ]
    for step in range(max_steps):
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:                 # no request → final answer
            return _text(ai.content)
        for tc in ai.tool_calls:              # run each requested tool
            print(f"[Step {step + 1}] Tool: {tc['name']}({tc['args']})")
            try:
                result = TOOL_FUNCTIONS[tc["name"]].invoke(tc["args"])
            except Exception as e:
                result = {"error": str(e)}
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"])) #ToolMessage is a message that contains the result of a tool call.
    return "Incomplete — max steps reached."


def main() -> None:
    q = "Why did D-110 drop? Any related ops notes?"
    print("Q:", q, "\n")
    print("\nA:", run_tool_loop(q))


if __name__ == "__main__":
    main()

"""
Q: Why did D-110 drop? Any related ops notes? 

[Step 1] Tool: rag_search({'query': 'D-110 drop'})

A: D-110 SDK activation dropped after rollout paused in 2 regions (source_id: ops_w25).
"""