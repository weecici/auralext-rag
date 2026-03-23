"""Text chunking with LangChain splitter + title generation via Google Generative AI."""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A chunk of text with its position metadata."""

    text: str
    index: int  # 0-based position within the source document
    source: str  # originating file path or identifier
    title: Optional[str] = None  # populated later by LLM


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Split *text* into overlapping chunks using LangChain's recursive splitter."""
    chunk_size = chunk_size or settings.MAX_TOKENS
    chunk_overlap = chunk_overlap or settings.OVERLAP_TOKENS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    return [TextChunk(text=c, index=i, source=source) for i, c in enumerate(raw_chunks)]


# ---------------------------------------------------------------------------
# Title generation via Google Generative AI
# ---------------------------------------------------------------------------


TITLE_SYSTEM_PROMPT = (
    f"You are a concise title generator. Given a text chunk from a document:\n"
    f"- Produce a short title with at most {settings.TITLE_MAX_TOKENS} tokens"
    f" that captures the main topic.\n"
    f"- The MAIN LANGUAGE of the title MUST BE THE SAME as the MAIN LANGUAGE"
    f" of the input text.\n"
    f"- Output ONLY the title, with no extra explanation or formatting."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "human" if settings.TITLE_GEN_MODEL.startswith("gemma-3") else "system",
            TITLE_SYSTEM_PROMPT,
        ),
        ("human", "{text}"),
    ]
)


@lru_cache(maxsize=1)
def _get_title_llm() -> ChatGoogleGenerativeAI:
    """Return a cached LLM client configured for title generation."""
    return ChatGoogleGenerativeAI(
        google_api_key=settings.GOOGLE_API_KEY,
        model=settings.TITLE_GEN_MODEL,
        temperature=settings.TITLE_GEN_TEMPERATURE,
        max_tokens=settings.TITLE_MAX_TOKENS,
    )


def _generate_title_sync(text: str) -> str | None:
    """Call Google Generative AI to generate a title for a chunk (blocking)."""
    llm = _get_title_llm()
    try:
        response = llm.invoke(
            PROMPT_TEMPLATE.format_messages(text=text),
        )
        title = (response.content or "").strip().strip("\"'")
        return title or None
    except Exception as exc:
        logger.warning(f"Title generation failed: {exc}")
        return None


async def generate_titles(chunks: list[TextChunk]) -> list[TextChunk]:
    """Generate titles for all chunks concurrently."""
    loop = asyncio.get_running_loop()

    async def _title_one(chunk: TextChunk) -> None:
        chunk.title = await loop.run_in_executor(None, _generate_title_sync, chunk.text)

    await asyncio.gather(*[_title_one(c) for c in chunks])
    return chunks
