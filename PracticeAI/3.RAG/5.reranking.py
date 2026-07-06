"""
Example E — Re-ranking
======================

Two-stage retrieval:
  Stage 1 (fast, approximate): vector DB returns the top-N "probably relevant"
                               candidates.
  Stage 2 (slow, precise):     a re-ranker scores each (query, chunk) PAIR and
                               keeps only the top few for the LLM.

Why: vector search ranks by embedding distance, which is fast but coarse. A
cross-encoder re-ranker reads the query and chunk TOGETHER, so it judges
relevance far more accurately — at the cost of an extra step.

    query → vector DB: top 20 candidates (fast)
          → re-ranker: score each query-chunk pair (accurate)
          → top 3 → LLM

The reference example used Cohere's hosted reranker (needs a paid API key). This
file uses `flashrank` instead — a free, local, lightweight cross-encoder (small
ONNX model, no key, no GPU). The Cohere version is shown at the bottom for
reference.

Why isn't vector search enough?

Suppose your database contains these chunks:

Chunk	Text
A	KPI D-110: activations fell to 12k.
B	Weekly install-to-activate funnel declined sharply.
C	SDK funnel commitment status is amber.
D	IM3 network cleanliness improved.

The user asks:

Why did D-110 activation drop?

The vector database converts the query into an embedding.

It also has embeddings for every chunk.

It compares vectors using cosine similarity.

Suppose it returns

Rank	Chunk
1	B
2	A
3	C
4	D

Notice something.

Chunk B doesn't mention D-110.

It only talks about activation decline.

The embedding model thinks

activation drop
≈
activation declined

So it ranks it highly.

But what does the user actually want?

The user asked

Why did D-110 activation drop?

There are two conditions:

D-110
activation drop

Chunk A satisfies both.

Chunk B satisfies only one.

Humans would clearly rank A above B.

Embedding similarity sometimes cannot.

Stage 1: Fast Retrieval

The vector DB only tries to answer

"Which chunks are probably relevant?"

It doesn't spend much computation.

User Query
      │
      ▼
Vector Search
      │
      ▼
Top 20 Candidates

Maybe

1 B
2 A
3 C
4 D
...
20
Stage 2: Re-ranking

Now comes the smarter model.

Instead of comparing vectors,

it reads

Query

AND

Chunk

together.

For every candidate.

Example:

Pair 1
Query:
Why did D-110 activation drop?

Chunk:
Weekly install-to-activate funnel declined sharply.

The model thinks

Activation ✔

D-110 ✘

Score = 0.63
Pair 2
Query:
Why did D-110 activation drop?

Chunk:
KPI D-110:
activations fell to 12k.

Now it sees

D-110 ✔

Activation ✔

Fell ≈ drop ✔

Score = 0.98

Much higher.

Pair 3
Query:
Why did D-110 activation drop?

Chunk:
SDK funnel commitment status amber.
D-110 ✘

Activation ✘

Score = 0.05

Final ranking becomes

Chunk	Score
A	0.98
B	0.63
C	0.05

Now only

A
B

go to the LLM.

Why is this more accurate?

The vector database computes

Embedding(query)

Embedding(chunk)

independently.

Query
   │
Embedding
   │

Chunk
   │
Embedding

It never compares individual words directly.

A cross-encoder does something different.

It feeds both texts into the same transformer:

[CLS] Classification token is a special token that is used to indicate the start of a sequence.

Query

[SEP] Separator token is a special token that is used to separate the query and the chunk.

Chunk

The transformer can now learn relationships like

"D-110"

appears

inside

this chunk

or

drop

means

fell

or

question asks WHY

chunk only contains statistics

It understands interactions between the query and the document that separate embeddings cannot capture.

Why not use the cross-encoder on every document?

Imagine

1 million chunks

A cross-encoder must process

Query + Chunk1

Query + Chunk2

Query + Chunk3

...

Query + Chunk1000000

That would be extremely slow.

Instead:

1 million chunks
       │
       ▼
Vector Search
       │
Top 20
       │
       ▼
Cross Encoder
       │
Top 3
       ▼
LLM

The expensive model only evaluates a small shortlist, making the system both fast and accurate.

What is FlashRank?

FlashRank is a lightweight local re-ranker.

Instead of calling a paid API, it downloads a small ONNX model and runs it on your own machine.

Query

+

Chunk

↓

FlashRank

↓

0.91

No internet.

No API key.

No GPU required.

Cohere ReRank vs FlashRank
Feature	FlashRank	Cohere ReRank
Runs locally	✔	✘
API key needed	✘	✔
Cost	Free	Paid
Speed	Fast	Fast
Accuracy	Good	Generally higher
Best for	Development, demos, local apps	Production systems where maximum ranking quality is needed
Summary

Think of two-stage retrieval like hiring:

Stage 1 (Vector Search): A recruiter quickly scans thousands of résumés and selects the top 20 candidates based on broad relevance.
Stage 2 (Re-ranker): A hiring manager carefully reviews those 20 candidates against the job description and chooses the best 3.



Setup:
    ../1.KPI_Narrator/.venv/bin/pip install flashrank
    ../1.KPI_Narrator/.venv/bin/python 5.reranking.py
    # First run downloads a small (~4MB) model; needs internet once.
"""

from flashrank import RerankRequest, Ranker

# Pretend these are the top-N candidates a vector DB already returned (Stage 1).
# Deliberately noisy: several mention "SDK" or "funnel" but only some are truly
# about the COMMITMENT STATUS the query asks for.
CANDIDATES = [
    {"id": "C-09", "text": "Commitment C-09: SDK funnel tracking — status amber, owner Product."},
    {"id": "D-110", "text": "KPI D-110: SDK activations fell to 12k from 24k prior week."},
    {"id": "ops_w25", "text": "Ops note: SDK rollout paused in 2 regions due to pack coverage gap."},
    {"id": "D-150", "text": "Billing feed D-150: ARPU IDR 44,000 for the SDK cohort."},
    {"id": "hr_2026", "text": "HR: the 2026 company holiday calendar was published."},
]


def main() -> None:
    query = "SDK funnel commitment status"

    # Stage 2: re-rank the candidates by true query-chunk relevance.
    ranker = Ranker()  # default small model; downloads once, then cached
    reranked = ranker.rerank(RerankRequest(query=query, passages=CANDIDATES))

    print("Q:", query, "\n")
    print("Re-ranked candidates (higher score = more relevant):")
    for r in reranked:
        print(f"  {r['score']:.4f} | {r['id']:>8} | {r['text']}")

    top_3 = [r["id"] for r in reranked[:3]]
    print(f"\n-> Top 3 sent to the LLM: {top_3}")
    print(
        "\nThe commitment-status line (C-09) should rank first, above chunks that\n"
        "merely contain 'SDK'. The HR holiday line sinks to the bottom. That\n"
        "precision — reading query and chunk together — is what a re-ranker buys.\n"
        "Tradeoff: an extra step and some latency, for materially better context."
    )


# --- Reference: the production Cohere version (needs COHERE_API_KEY) ---------
# import cohere
# co = cohere.Client()  # reads COHERE_API_KEY
# rerank = co.rerank(
#     model="rerank-english-v3.0",
#     query=query,
#     documents=[c["text"] for c in CANDIDATES],
#     top_n=3,
# )
# best_chunks = [CANDIDATES[r.index]["text"] for r in rerank.results]


if __name__ == "__main__":
    main()
