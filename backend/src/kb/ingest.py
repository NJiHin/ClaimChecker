"""PDF ingestion pipeline with contextual embeddings.

Chunks a PDF into passages, enriches each chunk with a situating sentence
via Gemini 2.0 Flash (contextual embeddings), embeds the enriched text with
gemini-embedding-001, and stores everything in the passages table.
"""

from __future__ import annotations

import hashlib
import io
import time
import uuid

import asyncpg
import pypdf
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CONTEXT_PROMPT = """\
Here is a chunk from the document above:
<chunk>
{chunk_text}
</chunk>

Write a single sentence (maximum 75 words) that situates this chunk within the \
overall document. State the document topic, the section it belongs to, and what \
this specific chunk covers. Do not restate the chunk text verbatim."""

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _extract_text(pdf_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk_text(text: str) -> list[str]:
    return [c for c in _splitter.split_text(text) if c.strip()]


# ---------------------------------------------------------------------------
# Contextual enrichment
# ---------------------------------------------------------------------------

def _enrich_passages(
    client: genai.Client,
    full_text: str,
    chunks: list[str],
) -> list[str]:
    """Return enriched texts (context sentence + chunk) for each chunk.

    Calls Gemini 2.0 Flash sequentially so the shared document prefix is
    cached after the first request.
    """
    doc_prefix = f"<document>\n{full_text}\n</document>"
    enriched: list[str] = []

    for chunk in chunks:
        prompt = _CONTEXT_PROMPT.format(chunk_text=chunk)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[doc_prefix, prompt],
            config=genai.types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.0,
            ),
        )
        context_sentence = response.text.strip() if response.text else ""
        enriched.append(f"{context_sentence}\n\n{chunk}" if context_sentence else chunk)

    return enriched


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in result.embeddings]


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

async def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    conn: asyncpg.Connection,
    client: genai.Client
) -> int:
    """Ingest a PDF into the passages table. Returns the number of passages stored."""
    full_text = _extract_text(pdf_bytes)
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    enriched = _enrich_passages(client, full_text, chunks)
    embeddings = _embed_texts(client, enriched)

    doc_title = filename.removesuffix(".pdf")
    doc_id = hashlib.sha256(f"{doc_title}:{full_text[:200]}".encode()).hexdigest()[:16]
    now = int(time.time() * 1000)
    
    rows = [
        (
            str(uuid.uuid4()),  # id
            doc_id,             # doc_id
            doc_title,          # doc_title
            chunk,              # text (raw)
            enriched[i],        # context_text
            str(embeddings[i]),  # embedding (cast to string for ::vector)
            None,               # page_start
            now,                # created_at
        )
        for i, chunk in enumerate(chunks)
    ]

    await conn.executemany(
        """
        INSERT INTO passages (id, doc_id, doc_title, text, context_text, embedding, page_start, created_at)
        VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8)
        ON CONFLICT (id) DO NOTHING
        """,
        rows,
    )

    return len(rows)
