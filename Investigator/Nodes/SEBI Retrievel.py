#!/usr/bin/env python3
"""
RAG Query Pipeline
==================
Given a natural-language question:
  1. Embed the question with OpenAI
  2. Query Pinecone for the top-K most similar chunks
  3. Build a prompt with those chunks as context
  4. Ask GPT-4o to answer using only the retrieved context

Run:
    python query.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ── Config (must match embed_and_upsert.py) ───────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "sk-...")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "pcsk_...")
PINECONE_INDEX_NAME = "sebi-regulations"
EMBEDDING_MODEL     = "text-embedding-3-small"
CHAT_MODEL          = "gpt-4o"
TOP_K               = 5      # number of chunks to retrieve


# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
index         = pc.Index(PINECONE_INDEX_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(question: str, top_k: int = TOP_K, filter: dict | None = None) -> list[dict]:
    """
    Embed the question and retrieve the top-K closest chunks from Pinecone.

    `filter` is an optional Pinecone metadata filter, e.g.:
        {"source": {"$eq": "SEBI_Mutual_Funds_Regulations_2026"}}
        {"page":   {"$gte": 1, "$lte": 5}}

    Returns a list of match dicts, each with:
        {
            "id":       chunk SHA-256,
            "score":    cosine similarity (0.0–1.0),
            "metadata": { source, page, chunk_index, ... }
        }
    """
    embedding = openai_client.embeddings.create(
        input = [question],
        model = EMBEDDING_MODEL,
    ).data[0].embedding

    results = index.query(
        vector           = embedding,
        top_k            = top_k,
        include_metadata = True,
        filter           = filter,
    )
    return results.matches


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a financial regulation expert assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Always cite the source document and page number for each claim."""


def answer(question: str, matches: list[dict]) -> str:
    """
    Build a RAG prompt from retrieved chunks and generate an answer.

    Context block format:
        [1] Source: <filename> | Page: <n>
        <chunk text>

    The model is instructed to cite [1], [2] etc. in its response.
    """
    context_parts = []
    for i, match in enumerate(matches, start=1):
        meta = match.metadata
        text = meta.get("chunk_text", "")  # text stored in metadata

        # Note: in embed_and_upsert.py we store text inside metadata
        # If you stored it separately, fetch it from your JSONL files instead
        context_parts.append(
            f"[{i}] Source: {meta.get('source', 'Unknown')} | "
            f"Page: {meta.get('page', '?')} | "
            f"Score: {match.score:.3f}\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    response = openai_client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = messages,
        temperature = 0,        # deterministic for regulatory Q&A
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE QUERY LOOP
# ══════════════════════════════════════════════════════════════════════════════

def query(
    question: str,
    top_k: int = TOP_K,
    filter: dict | None = None,
    show_sources: bool = True,
) -> str:
    """
    Full RAG pipeline: question → embed → retrieve → generate → answer.

    Args:
        question     : Natural language question
        top_k        : Number of chunks to retrieve (default 5)
        filter       : Optional Pinecone metadata filter
        show_sources : Print retrieved source citations

    Returns:
        Generated answer string
    """
    matches = retrieve(question, top_k=top_k, filter=filter)

    if show_sources:
        print("\n── Retrieved Chunks ──────────────────────────────────────")
        for i, m in enumerate(matches, 1):
            meta = m.metadata
            print(
                f"  [{i}] {meta.get('source', '?')}  "
                f"page {meta.get('page', '?')}  "
                f"score={m.score:.3f}"
            )
        print("──────────────────────────────────────────────────────────\n")

    return answer(question, matches)


# ── Usage examples ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Basic question
    q = "What are the eligibility criteria for a Category II AIF?"
    print(f"Q: {q}\n")
    print(query(q))

    # Filtered by a specific regulation document
    print("\n" + "=" * 60 + "\n")
    q2 = "What are the net worth requirements for investment advisers?"
    print(f"Q: {q2}\n")
    print(query(
        q2,
        filter={"source": {"$eq": "SEBI_Investment_Advisers_Regulations_2013"}},
    ))