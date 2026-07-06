"""
Example B — Chunking strategies
===============================

Chunking = splitting a large document into pieces small enough to embed and
retrieve well. Chunk quality caps retrieval quality: you can only retrieve a
chunk that was worth storing on its own.

No API key needed — this is pure Python string work.

Strategy            | When to use
--------------------|------------------------------------------
Fixed size          | Generic docs, logs
Paragraph / section | QBR, Confluence, reports
Semantic chunking   | Split where meaning shifts (advanced)
One row per chunk    | KPI sheets, CSV rows

Typical starting size: 300-800 tokens per chunk, 10-20% overlap.
  Too small -> loses context ("it" refers to nothing).
  Too large -> diluted embedding, retrieves the wrong section.

Run:
    ../1.KPI_Narrator/.venv/bin/python 2.chunking.py
"""


def chunk_by_chars(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Simple character-based chunking with overlap.

    `overlap` re-includes the last few characters of the previous chunk at the
    start of the next one, so a sentence split across a boundary still appears
    whole in at least one chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step back by `overlap` before the next chunk
    return chunks


def chunk_by_paragraphs(text: str) -> list[str]:
    """Better for structured docs (QBR, reports) — split on blank lines so each
    chunk is a coherent section, not an arbitrary character window."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# Example QBR text (note the blank line between the two sections).
qbr_text = """
## SDK Commitments
We committed to delivering full install-to-churn funnel visibility by Q3 2026.
Owner: Product team. Status: at risk due to missing D-110 events.

## ARPU Impact
SDK-enabled cohort showed ARPU uplift of 17% vs control group.
Billing feed D-150 confirms IDR 44,000 average.
"""


def main() -> None:
    print("=== chunk_by_paragraphs (section-aware) ===\n")
    para_chunks = chunk_by_paragraphs(qbr_text)
    print("para_chunks: ", para_chunks)
    for i, c in enumerate(para_chunks):
        print(f"--- Chunk {i} ---\n{c}\n")

    print("=== chunk_by_chars (fixed 120 chars, 20 overlap) ===\n")
    char_chunks = chunk_by_chars(qbr_text.strip(), chunk_size=120, overlap=20)
    print("char_chunks: ", char_chunks)
    for i, c in enumerate(char_chunks):
        # repr() shows the boundaries and the overlap clearly
        print(f"--- Chunk {i} ({len(c)} chars) ---\n{c!r}\n")

    print(
        "Notice: paragraph chunks are clean sections you'd be happy to retrieve.\n"
        "Fixed-size chunks cut mid-sentence — the overlap is what keeps a split\n"
        "sentence readable in at least one chunk. Match the strategy to the doc."
    )


if __name__ == "__main__":
    main()
