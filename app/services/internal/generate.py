"""Internal service: LLM text generation via Google Generative AI (LangChain).

Provides both **non-streaming** and **streaming** generation using
``langchain_google_genai.ChatGoogleGenerativeAI``.

All synchronous SDK calls are wrapped for ``asyncio.run_in_executor`` so the
event loop is never blocked.
"""

import asyncio
from functools import lru_cache
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.logging import logger


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    """Return a cached Google Generative AI client (shared for all generation)."""
    return ChatGoogleGenerativeAI(
        google_api_key=settings.GOOGLE_API_KEY,
        model=settings.GENERATION_MODEL,
        temperature=settings.GENERATION_TEMPERATURE,
        max_tokens=settings.GENERATION_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# RAG system prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable assistant. Answer the user's question based on the \
provided context documents. Follow these rules:

1. Use ONLY the information from the context to answer. If the context does \
not contain enough information, say so honestly.
2. Be concise and direct. Avoid unnecessary filler.
3. If the user asks a follow-up question, use conversation history for context \
but always ground answers in the retrieved documents.
4. Answer in the same language as the user's question.
"""


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def build_context_block(sources: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the LLM prompt.

    Each source dict should have at least ``text`` and optionally ``title``
    and ``score``.
    """
    if not sources:
        return "(No relevant documents found.)"

    parts: list[str] = []
    for i, src in enumerate(sources, 1):
        title = src.get("title") or "Untitled"
        text = src.get("text", "")
        score = src.get("score")
        header = f"[Document {i}: {title}]"
        if score is not None:
            header += f" (relevance: {score:.2f})"
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def build_messages(
    *,
    user_query: str,
    context_block: str,
    history: list[dict[str, str]],
) -> list[BaseMessage]:
    """Assemble LangChain messages for a RAG turn.

    Layout:
    1. **SystemMessage**: RAG instruction prompt
    2. **History**: previous (Human / AI) turns (trimmed to N turns)
    3. **HumanMessage**: context block + current question

    The context is injected in the *latest user turn* rather than the system
    prompt so the model treats it as grounding material for the current
    question, not as a persistent instruction that might leak across turns.
    """
    messages: list[BaseMessage] = [
        (
            HumanMessage(content=RAG_SYSTEM_PROMPT)
            if settings.GENERATION_MODEL.startswith("gemma-3")
            else SystemMessage(content=RAG_SYSTEM_PROMPT)
        )
    ]

    _role_map: dict[str, type[BaseMessage]] = {
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    for h in history:
        cls = _role_map.get(h["role"], HumanMessage)
        if cls:
            messages.append(cls(content=h["content"]))

    user_content = (
        f"Context documents:\n\n{context_block}\n\n---\n\nQuestion: {user_query}"
    )
    messages.append(HumanMessage(content=user_content))
    return messages


# ---------------------------------------------------------------------------
# Synchronous helpers (called via run_in_executor)
# ---------------------------------------------------------------------------


def _generate_sync(messages: list[BaseMessage]) -> str:
    """Blocking call to Google Generative AI. Returns the full response text."""
    llm = _get_llm()
    response = llm.invoke(messages)
    return response.content


def _generate_stream_sync(messages: list[BaseMessage]) -> list[str]:
    """Blocking streaming call. Returns an iterable of content delta strings."""
    llm = _get_llm()
    return [chunk.content for chunk in llm.stream(messages) if chunk.content]


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


async def generate(messages: list[BaseMessage]) -> str:
    """Generate a complete response (non-streaming).

    Offloads the blocking LangChain call to the default thread-pool executor
    so the event loop stays free.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generate_sync, messages)


async def generate_stream(
    messages: list[BaseMessage],
) -> asyncio.Queue[str | None]:
    """Start a streaming generation and return an ``asyncio.Queue``.

    The caller reads tokens from the queue.  A ``None`` sentinel signals
    end-of-stream.  The actual blocking iteration runs in a thread-pool
    executor.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _producer() -> None:
        try:
            llm = _get_llm()
            for chunk in llm.stream(messages):
                if chunk.content:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.content)
        except Exception as exc:
            logger.error(f"Streaming generation error: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _producer)
    return queue
