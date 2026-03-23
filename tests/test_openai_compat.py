"""Tests for the OpenAI-compatible endpoints (Open WebUI integration).

Covers:
- GET /api/v1/models — lists Milvus collections as models
- POST /api/v1/chat/completions — non-streaming RAG chat
- POST /api/v1/chat/completions — streaming SSE in OpenAI format
- Helper functions: _collection_from_model, _trim_openai_history, etc.
- Error handling: unknown model, missing user message, search/generation failures
- Concurrency: multiple requests served in parallel

All external services (Milvus, search, generate) are mocked.
"""

import asyncio
import json
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.search import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _make_search_results(n: int = 2) -> list[SearchResult]:
    return [
        SearchResult(
            doc_id=i,
            title=f"Document {i}",
            text=f"Content of document {i}.",
            score=0.9 - i * 0.1,
        )
        for i in range(1, n + 1)
    ]


# ===================================================================
# 1. Helper function unit tests
# ===================================================================


class TestCollectionFromModel:
    """Test _collection_from_model helper."""

    def test_valid_model_id(self):
        from app.services.public.openai_compat import _collection_from_model

        assert _collection_from_model("RAG_KB/my_docs") == "my_docs"

    def test_model_with_hyphens(self):
        from app.services.public.openai_compat import _collection_from_model

        assert _collection_from_model("RAG_KB/cs431-lectures") == "cs431-lectures"

    def test_model_with_underscores(self):
        from app.services.public.openai_compat import _collection_from_model

        assert _collection_from_model("RAG_KB/ml_papers_2026") == "ml_papers_2026"

    def test_invalid_prefix(self):
        from app.services.public.openai_compat import _collection_from_model

        assert _collection_from_model("gpt-4") is None

    def test_empty_string(self):
        from app.services.public.openai_compat import _collection_from_model

        assert _collection_from_model("") is None

    def test_just_prefix(self):
        from app.services.public.openai_compat import _collection_from_model

        assert _collection_from_model("RAG_KB/") == ""


class TestListDocumentCollections:
    """Test _list_document_collections helper."""

    def test_filters_internal_collections(self):
        from app.services.public.openai_compat import _list_document_collections

        mock_client = MagicMock()
        mock_client.list_collections.return_value = [
            "my_docs",
            "_conversation_meta",
            "_conversation_messages",
            "lectures",
        ]

        with patch(
            "app.services.public.openai_compat.get_client",
            return_value=mock_client,
        ):
            result = _list_document_collections()

        assert result == ["lectures", "my_docs"]  # sorted, no _ prefix

    def test_empty_collections(self):
        from app.services.public.openai_compat import _list_document_collections

        mock_client = MagicMock()
        mock_client.list_collections.return_value = []

        with patch(
            "app.services.public.openai_compat.get_client",
            return_value=mock_client,
        ):
            result = _list_document_collections()

        assert result == []

    def test_only_internal_collections(self):
        from app.services.public.openai_compat import _list_document_collections

        mock_client = MagicMock()
        mock_client.list_collections.return_value = [
            "_conversation_meta",
            "_conversation_messages",
        ]

        with patch(
            "app.services.public.openai_compat.get_client",
            return_value=mock_client,
        ):
            result = _list_document_collections()

        assert result == []


class TestTrimOpenaiHistory:
    """Test _trim_openai_history helper."""

    def test_extracts_history_excluding_last_user(self):
        from app.services.public.openai_compat import _trim_openai_history, ChatMessage

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Current question"),
        ]
        result = _trim_openai_history(messages, max_turns=5)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "First question"}
        assert result[1] == {"role": "assistant", "content": "First answer"}

    def test_no_history(self):
        from app.services.public.openai_compat import _trim_openai_history, ChatMessage

        messages = [
            ChatMessage(role="system", content="System"),
            ChatMessage(role="user", content="Only question"),
        ]
        result = _trim_openai_history(messages, max_turns=5)
        assert result == []

    def test_trims_to_max_turns(self):
        from app.services.public.openai_compat import _trim_openai_history, ChatMessage

        messages = [
            ChatMessage(role="user", content="Q1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="Q2"),
            ChatMessage(role="assistant", content="A2"),
            ChatMessage(role="user", content="Q3"),  # current
        ]
        result = _trim_openai_history(messages, max_turns=1)
        assert len(result) == 2
        assert result[0]["content"] == "Q2"
        assert result[1]["content"] == "A2"

    def test_empty_messages(self):
        from app.services.public.openai_compat import _trim_openai_history, ChatMessage

        result = _trim_openai_history([], max_turns=5)
        assert result == []

    def test_system_only(self):
        from app.services.public.openai_compat import _trim_openai_history, ChatMessage

        messages = [ChatMessage(role="system", content="System")]
        result = _trim_openai_history(messages, max_turns=5)
        assert result == []


class TestBuildNonStreamingResponse:
    """Test _build_non_streaming_response helper."""

    def test_response_shape(self):
        from app.services.public.openai_compat import _build_non_streaming_response

        result = _build_non_streaming_response(
            "chatcmpl-abc", "RAG_KB/docs", "Hello world"
        )
        assert result["id"] == "chatcmpl-abc"
        assert result["object"] == "chat.completion"
        assert result["model"] == "RAG_KB/docs"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "usage" in result


class TestBuildStreamingChunk:
    """Test _build_streaming_chunk helper."""

    def test_content_chunk(self):
        from app.services.public.openai_compat import _build_streaming_chunk

        result = _build_streaming_chunk("id-1", "model", content="Hello")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_finish_chunk(self):
        from app.services.public.openai_compat import _build_streaming_chunk

        result = _build_streaming_chunk("id-1", "model", finish_reason="stop")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["delta"] == {}
        assert data["choices"][0]["finish_reason"] == "stop"


# ===================================================================
# 2. GET /api/v1/models endpoint tests
# ===================================================================


class TestModelsEndpoint:
    """GET /api/v1/models"""

    def test_list_models_200(self, client: TestClient):
        with patch(
            "app.services.public.openai_compat._list_document_collections",
            return_value=["cs431_lectures", "ml_papers"],
        ):
            response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "RAG_KB/cs431_lectures"
        assert data["data"][1]["id"] == "RAG_KB/ml_papers"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "auralext-rag"

    def test_list_models_empty(self, client: TestClient):
        with patch(
            "app.services.public.openai_compat._list_document_collections",
            return_value=[],
        ):
            response = client.get("/api/v1/models")

        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_list_models_single(self, client: TestClient):
        with patch(
            "app.services.public.openai_compat._list_document_collections",
            return_value=["my_docs"],
        ):
            response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "RAG_KB/my_docs"


# ===================================================================
# 3. POST /api/v1/chat/completions — non-streaming tests
# ===================================================================


class TestChatCompletionsNonStreaming:
    """POST /api/v1/chat/completions with stream=false"""

    def test_basic_completion_200(self, client: TestClient):
        search_results = _make_search_results(2)

        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=search_results,
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="AI stands for Artificial Intelligence.",
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/my_docs",
                    "messages": [
                        {"role": "user", "content": "What is AI?"},
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "RAG_KB/my_docs"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert (
            data["choices"][0]["message"]["content"]
            == "AI stands for Artificial Intelligence."
        )
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_unknown_model_404(self, client: TestClient):
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "model_not_found" in data["error"]["code"]

    def test_no_user_message_400(self, client: TestClient):
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "RAG_KB/docs",
                "messages": [{"role": "system", "content": "You are helpful"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_with_system_and_history(self, client: TestClient):
        """Messages with system prompt and history are handled correctly."""
        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Follow-up answer",
            ) as mock_gen,
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/my_docs",
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "First question"},
                        {"role": "assistant", "content": "First answer"},
                        {"role": "user", "content": "Follow-up"},
                    ],
                },
            )

        assert response.status_code == 200
        # The generate function should have been called with messages
        # that include history
        mock_gen.assert_awaited_once()
        llm_messages = mock_gen.call_args[0][0]
        # System + history (2 msgs) + current user = 4
        assert len(llm_messages) == 4

    def test_search_uses_last_user_message(self, client: TestClient):
        """Search is performed using the last user message."""
        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_search,
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
        ):
            client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [
                        {"role": "user", "content": "Old question"},
                        {"role": "assistant", "content": "Old answer"},
                        {"role": "user", "content": "New question"},
                    ],
                },
            )

        mock_search.assert_awaited_once()
        assert mock_search.call_args.kwargs["query"] == "New question"
        assert mock_search.call_args.kwargs["collection_name"] == "col"

    def test_empty_search_results_still_works(self, client: TestClient):
        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="No docs found, sorry.",
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/empty_col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert response.status_code == 200
        assert (
            response.json()["choices"][0]["message"]["content"]
            == "No docs found, sorry."
        )

    def test_search_failure_returns_500(self, client: TestClient):
        with patch(
            "app.services.public.openai_compat.search_documents",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Milvus connection lost"),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert response.status_code == 500
        assert "error" in response.json()

    def test_generation_failure_returns_500(self, client: TestClient):
        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Cerebras API error"),
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert response.status_code == 500
        assert "error" in response.json()

    def test_response_has_usage_field(self, client: TestClient):
        """OpenAI response must include usage field (even if zeroed)."""
        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        data = response.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]


# ===================================================================
# 4. POST /api/v1/chat/completions — streaming tests
# ===================================================================


class TestChatCompletionsStreaming:
    """POST /api/v1/chat/completions with stream=true"""

    def test_streaming_returns_sse(self, client: TestClient):
        """Streaming should return text/event-stream with OpenAI chunk format."""

        async def fake_generate_stream(messages):
            q: asyncio.Queue[str | None] = asyncio.Queue()
            q.put_nowait("Hello")
            q.put_nowait(" world")
            q.put_nowait(None)
            return q

        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate_stream",
                side_effect=fake_generate_stream,
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE lines
        body = response.text
        lines = [l for l in body.split("\n") if l.startswith("data: ")]

        # Should have: 2 content chunks + 1 finish chunk + [DONE]
        assert len(lines) >= 3

        # First chunk: content "Hello"
        chunk1 = json.loads(lines[0][6:])
        assert chunk1["choices"][0]["delta"]["content"] == "Hello"

        # Second chunk: content " world"
        chunk2 = json.loads(lines[1][6:])
        assert chunk2["choices"][0]["delta"]["content"] == " world"

        # Finish chunk
        finish = json.loads(lines[2][6:])
        assert finish["choices"][0]["finish_reason"] == "stop"

        # [DONE] terminator
        assert "data: [DONE]" in body

    def test_streaming_model_in_chunks(self, client: TestClient):
        """Each chunk should include the model ID."""

        async def fake_stream(messages):
            q: asyncio.Queue[str | None] = asyncio.Queue()
            q.put_nowait("token")
            q.put_nowait(None)
            return q

        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate_stream",
                side_effect=fake_stream,
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
            )

        lines = [
            l
            for l in response.text.split("\n")
            if l.startswith("data: ") and l != "data: [DONE]"
        ]
        for line in lines:
            chunk = json.loads(line[6:])
            assert chunk["model"] == "RAG_KB/col"
            assert chunk["object"] == "chat.completion.chunk"

    def test_streaming_unknown_model_404(self, client: TestClient):
        """Unknown model returns error even for streaming requests."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        # Error happens before streaming starts, so it's a normal JSON error
        assert response.status_code == 404

    def test_streaming_no_user_message_400(self, client: TestClient):
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "RAG_KB/col",
                "messages": [{"role": "system", "content": "system"}],
                "stream": True,
            },
        )

        assert response.status_code == 400

    def test_streaming_consistent_completion_id(self, client: TestClient):
        """All chunks in a stream should share the same completion ID."""

        async def fake_stream(messages):
            q: asyncio.Queue[str | None] = asyncio.Queue()
            q.put_nowait("a")
            q.put_nowait("b")
            q.put_nowait(None)
            return q

        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate_stream",
                side_effect=fake_stream,
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "q"}],
                    "stream": True,
                },
            )

        lines = [
            l
            for l in response.text.split("\n")
            if l.startswith("data: ") and l != "data: [DONE]"
        ]
        ids = set()
        for line in lines:
            chunk = json.loads(line[6:])
            ids.add(chunk["id"])

        assert len(ids) == 1  # All chunks share the same ID
        assert list(ids)[0].startswith("chatcmpl-")


# ===================================================================
# 5. Integration / end-to-end flow tests
# ===================================================================


class TestEndToEndFlow:
    """Test the full RAG flow through OpenAI-compat endpoints."""

    def test_rag_context_reaches_llm(self, client: TestClient):
        """Verify search results are formatted into the LLM prompt."""
        search_results = _make_search_results(2)
        captured_messages = []

        async def capture_generate(messages):
            captured_messages.extend(messages)
            return "Answer with context"

        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=search_results,
            ),
            patch(
                "app.services.public.openai_compat.generate",
                side_effect=capture_generate,
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "What is AI?"}],
                },
            )

        assert response.status_code == 200

        # The last message should contain context block + question
        last_msg = captured_messages[-1]
        assert "Document 1" in last_msg.content
        assert "Document 2" in last_msg.content
        assert "What is AI?" in last_msg.content

    def test_stream_false_is_default(self, client: TestClient):
        """Default stream=false returns JSON, not SSE."""
        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
        ):
            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


# ===================================================================
# 6. Concurrency tests
# ===================================================================


class TestOpenAIConcurrency:
    """Verify OpenAI-compat endpoints handle concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_completions(self):
        """Multiple non-streaming completions can run in parallel."""
        from app.services.public.openai_compat import (
            chat_completions,
            ChatCompletionRequest,
            ChatMessage,
        )

        req = ChatCompletionRequest(
            model="RAG_KB/col",
            messages=[ChatMessage(role="user", content="test")],
        )

        with (
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
        ):
            results = await asyncio.gather(
                chat_completions(req),
                chat_completions(req),
            )

        assert len(results) == 2
        # Both should have different completion IDs
        assert results[0]["id"] != results[1]["id"]


# ===================================================================
# 7. Schema validation tests
# ===================================================================


class TestOpenAISchemas:
    """Test the OpenAI-compat Pydantic models."""

    def test_chat_message_defaults(self):
        from app.services.public.openai_compat import ChatMessage

        msg = ChatMessage()
        assert msg.role == "user"
        assert msg.content == ""

    def test_chat_completion_request_defaults(self):
        from app.services.public.openai_compat import (
            ChatCompletionRequest,
            ChatMessage,
        )

        req = ChatCompletionRequest(
            model="RAG_KB/col",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert req.stream is False
        assert req.temperature is None
        assert req.max_tokens is None

    def test_chat_completion_request_with_stream(self):
        from app.services.public.openai_compat import (
            ChatCompletionRequest,
            ChatMessage,
        )

        req = ChatCompletionRequest(
            model="RAG_KB/col",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
            temperature=0.5,
            max_tokens=100,
        )
        assert req.stream is True
        assert req.temperature == 0.5
        assert req.max_tokens == 100

    def test_model_object(self):
        from app.services.public.openai_compat import ModelObject

        m = ModelObject(id="RAG_KB/docs")
        assert m.id == "RAG_KB/docs"
        assert m.object == "model"
        assert m.owned_by == "auralext-rag"
        assert isinstance(m.created, int)

    def test_model_list_response(self):
        from app.services.public.openai_compat import ModelListResponse, ModelObject

        resp = ModelListResponse(
            data=[ModelObject(id="RAG_KB/a"), ModelObject(id="RAG_KB/b")]
        )
        assert resp.object == "list"
        assert len(resp.data) == 2


# ===================================================================
# 8. Reranking integration tests (OPENWEBUI_RERANKING_ENABLED)
# ===================================================================


class TestOpenWebUIReranking:
    """Test that OPENWEBUI_RERANKING_ENABLED controls rerank param in search."""

    def test_reranking_enabled_passes_rerank_true(self, client: TestClient):
        """When OPENWEBUI_RERANKING_ENABLED=True, search gets rerank=True."""
        with (
            patch(
                "app.services.public.openai_compat.settings",
            ) as mock_settings,
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_search,
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
        ):
            mock_settings.OPENWEBUI_RERANKING_ENABLED = True
            mock_settings.GENERATION_SEARCH_TYPE = "hybrid"
            mock_settings.GENERATION_RAG_TOP_K = 5
            mock_settings.GENERATION_HISTORY_TURNS = 5

            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert response.status_code == 200
        mock_search.assert_awaited_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["rerank"] is True

    def test_reranking_disabled_passes_rerank_false(self, client: TestClient):
        """When OPENWEBUI_RERANKING_ENABLED=False (default), search gets rerank=False."""
        with (
            patch(
                "app.services.public.openai_compat.settings",
            ) as mock_settings,
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_search,
            patch(
                "app.services.public.openai_compat.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
        ):
            mock_settings.OPENWEBUI_RERANKING_ENABLED = False
            mock_settings.GENERATION_SEARCH_TYPE = "hybrid"
            mock_settings.GENERATION_RAG_TOP_K = 5
            mock_settings.GENERATION_HISTORY_TURNS = 5

            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert response.status_code == 200
        mock_search.assert_awaited_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["rerank"] is False

    def test_reranking_enabled_streaming(self, client: TestClient):
        """Streaming with OPENWEBUI_RERANKING_ENABLED=True also passes rerank=True."""

        async def fake_generate_stream(messages):
            q: asyncio.Queue[str | None] = asyncio.Queue()
            q.put_nowait("token")
            q.put_nowait(None)
            return q

        with (
            patch(
                "app.services.public.openai_compat.settings",
            ) as mock_settings,
            patch(
                "app.services.public.openai_compat.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_search,
            patch(
                "app.services.public.openai_compat.generate_stream",
                side_effect=fake_generate_stream,
            ),
        ):
            mock_settings.OPENWEBUI_RERANKING_ENABLED = True
            mock_settings.GENERATION_SEARCH_TYPE = "hybrid"
            mock_settings.GENERATION_RAG_TOP_K = 5
            mock_settings.GENERATION_HISTORY_TURNS = 5

            response = client.post(
                "/api/v1/chat/completions",
                json={
                    "model": "RAG_KB/col",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        mock_search.assert_awaited_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["rerank"] is True
