"""
Example C — LangChain RAG chain (LCEL)
======================================

RAG (Retrieval-Augmented Generation) pipeline is built using LangChain Expression Language (LCEL).
The | operator simply means "pass the output of the left component as the input to the right component.


The same retrieve -> augment -> generate flow as Example A, but wired together
with LangChain's Expression Language (LCEL) using the `|` (pipe) operator. Each
`|` feeds the left side's output into the right side:

    {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt        # fill the template with context + question
        | llm           # generate
        | StrOutputParser()   # pull the plain string out of the LLM message

`RunnablePassthrough()` just forwards the original input (the question) unchanged
into the "question" slot while the retriever fills the "context" slot.


Question
    │
    ▼
+------------------------------+
| Create two inputs            |
|                              |
| context  <- retriever        |
|              │               |
|              ▼               |
|         format_docs          |
|                              |
| question <- original input   |
|           (Passthrough)      |
+------------------------------+
             │
             ▼
        prompt template
             │
             ▼
            LLM
             │
             ▼
     StrOutputParser
             │
             ▼
        Final Answer


eg: 

Step 1: User asks a question

Suppose the user asks:

What is LangChain?

This is the input to the chain.

Step 2: Split the input into two paths
{
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
}

The same question is used in two different ways.

Path 1: Context
retriever

Searches the vector database.

Suppose it returns:

Document 1:
LangChain is a framework for building LLM applications.

Document 2:
It supports RAG and AI agents.

Then

format_docs

turns the retrieved documents into one string.

For example:

LangChain is a framework for building LLM applications.

It supports RAG and AI agents.

This becomes

context =

LangChain is a framework for building LLM applications.

It supports RAG and AI agents.

Path 2: Question
RunnablePassthrough()

does nothing.

Input

What is LangChain?

Output

What is LangChain?

It simply forwards the original question.

So now we have

{
    "context": "...retrieved text...",
    "question": "What is LangChain?"
}
Step 3: Prompt

Now the prompt template is filled.

Example template:

PromptTemplate(

Use the context below to answer.

Context:
context

Question:
question

)

After filling:

Use the context below to answer.

Context:
LangChain is a framework for building LLM applications.

It supports RAG and AI agents.

Question:
What is LangChain?

This entire prompt goes to the LLM.

Step 4: LLM

The LLM generates:

LangChain is a framework used to build applications powered by large language models. It also provides tools for retrieval-augmented generation and AI agents.
Step 5: StrOutputParser

LLMs often return an object like:

AIMessage(
    content="LangChain is a framework..."
)

StrOutputParser() extracts just the text:

LangChain is a framework...
Why use RunnablePassthrough()?

Without it:

{
    "context": retriever | format_docs
}

Only the context would be passed to the prompt.

The prompt also expects:

question

There would be no value for it.

RunnablePassthrough() solves this by forwarding the original input unchanged.

Input Question
       │
       ├──────────────► RunnablePassthrough
       │                     │
       │                     ▼
       │              question="What is LangChain?"
       │
       ▼
 retriever
       │
       ▼
Retrieved Docs
       │
       ▼
format_docs
       │
       ▼
context="LangChain is ..."

These two outputs are combined into:

{
    "context": "LangChain is ...",
    "question": "What is LangChain?"
}
Why use LCEL?

Instead of writing code like:

docs = retriever.invoke(question)
context = format_docs(docs)

prompt_text = prompt.invoke({
    "context": context,
    "question": question
})

response = llm.invoke(prompt_text)

answer = StrOutputParser().invoke(response)

LCEL lets you express the same workflow declaratively:

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

This is shorter, easier to read, and each component remains reusable and composable.


Provider swapped from OpenAI to free Gemini; Chroma is the in-memory vector store.

Run:
    ../1.KPI_Narrator/.venv/bin/python 3.langchain_rag_chain.py
"""

import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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


def main() -> None:
    texts = [
        "Commitment C-09: SDK funnel tracking — status amber.",
        "KPI D-110: activations 12k vs 24k prior week.",
    ]
    metadatas = [{"source_id": "C-09"}, {"source_id": "D-110"}]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # In-memory Chroma (no persist_directory) — fine for a small demo.
    vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # k=2 means return 2 documents from the vector database

    prompt = ChatPromptTemplate.from_template(
        "Use ONLY this context:\n{context}\n\n"
        "Question: {question}\n"
        "Cite source_id. If unknown, say insufficient evidence."
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def format_docs(docs):
        # Turn retrieved Document objects into a single context string with cites.
        return "\n".join(
            f"- {d.page_content} (source: {d.metadata.get('source_id')})"
            for d in docs
        )
    # Chain the components together
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What is the status of SDK funnel commitment?"
    print("Q:", question)
    print("A:", rag_chain.invoke(question))


if __name__ == "__main__":
    main()
