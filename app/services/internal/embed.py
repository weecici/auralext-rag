"""Dense embedding via Google Gemini Embedding API.

Two task types are exposed:

- **RETRIEVAL_DOCUMENT** — used at *ingestion* time to embed document chunks.
- **RETRIEVAL_QUERY**   — used at *search* time to embed the user query.

Google Gemini produces better retrieval results when the correct task type
is provided, because it applies asymmetric projection internally.
"""

import asyncio
from functools import lru_cache
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logging import logger


# ---------------------------------------------------------------------------
# Clients (cached singletons — one per task type)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_document_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Client for **document** embeddings (ingestion)."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.EMBEDDING_DIM,
    )


@lru_cache(maxsize=1)
def _get_query_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Client for **query** embeddings (search)."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=settings.EMBEDDING_DIM,
    )


# ---------------------------------------------------------------------------
# Synchronous helpers (called via run_in_executor)
# ---------------------------------------------------------------------------


def _embed_batch_sync(
    texts: list[str], titles: Optional[list[str]] = None
) -> list[list[float]]:
    """Embed a batch of **document** texts synchronously."""
    client = _get_document_embedding_client()
    return client.embed_documents(texts=texts, titles=titles)


def _embed_query_sync(text: str) -> list[float]:
    """Embed a single **query** text synchronously."""
    client = _get_query_embedding_client()
    return client.embed_query(text)


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


async def dense_embed(
    texts: list[str], titles: Optional[list[str]] = None
) -> list[list[float]]:
    """Embed a list of *document* texts, batching as needed.

    Returns a list of float vectors, one per input text, each of
    dimension ``settings.EMBEDDING_DIM``.
    """
    if not texts:
        return []

    if titles and len(titles) != len(texts):
        logger.warning("Length of titles does not match texts; expanding for embedding")
        titles = titles + [None] * (len(texts) - len(titles))

    batch_size = settings.EMBEDDING_BATCH_SIZE
    loop = asyncio.get_running_loop()

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_titles = titles[start : start + batch_size] if titles else None
        logger.debug(
            f"Embedding batch {start // batch_size + 1} "
            f"({len(batch_texts)} texts, model={settings.EMBEDDING_MODEL})"
        )
        vectors = await loop.run_in_executor(
            None, _embed_batch_sync, batch_texts, batch_titles
        )
        all_vectors.extend(vectors)

    return all_vectors


async def embed_query(text: str) -> list[float]:
    """Embed a single *search query* using ``RETRIEVAL_QUERY`` task type.

    Returns a float vector of dimension ``settings.EMBEDDING_DIM``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_query_sync, text)
