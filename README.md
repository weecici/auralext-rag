# Audio2Text RAG

A production-grade **Retrieval-Augmented Generation** system that ingests text and audio documents, builds a searchable vector store, and serves multi-turn conversational AI with source-grounded answers. Designed with a clean layered architecture, async-first concurrency, GPU memory safety, and drop-in OpenAI API compatibility for [Open WebUI](https://github.com/open-webui/open-webui) integration.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Architecture Overview](#architecture-overview)
- [System Architecture Diagram](#system-architecture-diagram)
- [Project Structure](#project-structure)
- [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
  - [API Layer](#api-layer)
  - [Schema Layer](#schema-layer)
  - [Middleware Layer](#middleware-layer)
  - [Service Layer — Public](#service-layer--public)
  - [Service Layer — Internal](#service-layer--internal)
  - [Repository Layer](#repository-layer)
  - [Core Infrastructure](#core-infrastructure)
- [Key Data Flows](#key-data-flows)
  - [Document Ingestion Pipeline](#document-ingestion-pipeline)
  - [RAG Conversation Pipeline](#rag-conversation-pipeline)
  - [Search Pipeline with Reranking](#search-pipeline-with-reranking)
- [Concurrency and GPU Management](#concurrency-and-gpu-management)
- [Technology Stack](#technology-stack)
- [Infrastructure & Deployment](#infrastructure--deployment)
- [Configuration Reference](#configuration-reference)

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, for reranking and transcription)

This starts Milvus, Redis, and Open WebUI.

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   GOOGLE_API_KEY=...
```

### 3. Setup Infrastructure + Run the Backend

```bash
docker compose up -d

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

if you have `just` installed, just simply run:

```bash
just
```

The API documentation is available at `http://localhost:8000/docs`.

### 4. Run Tests (Optional)

```bash
uv run pytest
```

### 5. Use Open WebUI

Open WebUI will be available at `http://localhost:8080`. It is preconfigured to use the FastAPI backend as its OpenAI provider, so you can start a conversation right away!

---

## Architecture Overview

The system follows a strict **four-layer architecture** with unidirectional dependencies:

```
API Endpoints -> Public Services -> Internal Services -> Repositories
```

Each layer has a single responsibility:

| Layer                 | Responsibility                                         | Examples                             |
| --------------------- | ------------------------------------------------------ | ------------------------------------ |
| **API**               | HTTP routing, request validation, response formatting  | REST endpoints, OpenAI-compat API    |
| **Public Services**   | Business orchestration, workflow coordination          | Ingestion pipeline, RAG chat, search |
| **Internal Services** | Atomic capabilities (embedding, generation, reranking) | Gemini embedding + LLM, CrossEncoder |
| **Repositories**      | Data access and persistence                            | Milvus vector DB, Redis job store    |

Cross-cutting concerns (configuration, logging, GPU locking, error handling) live in `app/core/` and `app/middleware/`.

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph Clients
        WEBUI[Open WebUI<br/><i>OpenAI-compatible client</i>]
        REST[REST API Client]
    end

    subgraph API["API Layer"]
        OAI["/api/v1/models<br/>/api/v1/chat/completions<br/><i>OpenAI Compat</i>"]
        EP_FILES["/api/v1/files/{collection}<br/><i>File Upload</i>"]
        EP_SEARCH["/api/v1/search/{collection}<br/><i>Vector Search</i>"]
        EP_CONV["/api/v1/conversations<br/><i>RAG Chat + SSE Streaming</i>"]
        EP_JOBS["/api/v1/jobs/{job_id}<br/><i>Job Status Polling</i>"]
        EP_HEALTH["/api/v1/health<br/><i>Liveness & Readiness</i>"]
    end

    subgraph MW["Middleware"]
        AUTH[Auth Middleware]
        RL[Rate Limiter]
        RC[Request Context]
        ERR[Error Handlers]
    end

    subgraph PubSvc["Public Services"]
        SVC_INGEST[Ingestion Service<br/><i>Orchestrates file processing</i>]
        SVC_SEARCH[Search Service<br/><i>Dense / Sparse / Hybrid + Rerank</i>]
        SVC_CONV[Conversation Service<br/><i>RAG: Retrieve → Augment → Generate</i>]
        SVC_JOB[Job Status Service]
    end

    subgraph IntSvc["Internal Services"]
        CHUNK[Chunking<br/><i>RecursiveCharacterTextSplitter</i>]
        EMBED[Embedding<br/><i>Google Gemini</i>]
        GEN[Generation<br/><i>Google Gemma 3</i>]
        RERANK[Reranking<br/><i>CrossEncoder</i>]
        STT[Speech-to-Text<br/><i>faster-whisper</i>]
        PROC[File Processing<br/><i>Load → Chunk → Embed</i>]
    end

    subgraph Repos["Repository Layer"]
        MILVUS_SEARCH[Milvus Search<br/><i>Dense / Sparse / Hybrid</i>]
        MILVUS_STORE[Milvus Storage<br/><i>Upsert / Delete</i>]
        MILVUS_CONV[Milvus Conversations<br/><i>Meta + Messages</i>]
        REDIS[Redis Job Store<br/><i>Async Job Tracking</i>]
    end

    subgraph Infra["Infrastructure"]
        MILVUS_DB[(Milvus v2.6<br/>Vector Database)]
        REDIS_DB[(Redis 8<br/>Cache)]
        GPU{{GPU / CUDA<br/><i>Shared Lock</i>}}
    end

    WEBUI --> OAI
    REST --> EP_FILES & EP_SEARCH & EP_CONV & EP_JOBS & EP_HEALTH

    OAI --> SVC_SEARCH & GEN
    EP_FILES --> SVC_INGEST
    EP_SEARCH --> SVC_SEARCH
    EP_CONV --> SVC_CONV
    EP_JOBS --> SVC_JOB

    SVC_INGEST --> PROC & STT & REDIS
    SVC_SEARCH --> EMBED & RERANK & MILVUS_SEARCH
    SVC_CONV --> SVC_SEARCH & GEN & MILVUS_CONV
    SVC_JOB --> REDIS

    PROC --> CHUNK & EMBED
    PROC --> MILVUS_STORE

    RERANK -.->|gpu_lock| GPU
    STT -.->|gpu_lock| GPU

    MILVUS_SEARCH --> MILVUS_DB
    MILVUS_STORE --> MILVUS_DB
    MILVUS_CONV --> MILVUS_DB
    REDIS --> REDIS_DB

    classDef apiStyle fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef svcStyle fill:#50B86E,stroke:#2D7A42,color:#fff
    classDef intStyle fill:#F5A623,stroke:#C47D12,color:#fff
    classDef repoStyle fill:#9B59B6,stroke:#6C3483,color:#fff
    classDef infraStyle fill:#34495E,stroke:#1C2833,color:#fff

    class OAI,EP_FILES,EP_SEARCH,EP_CONV,EP_JOBS,EP_HEALTH apiStyle
    class SVC_INGEST,SVC_SEARCH,SVC_CONV,SVC_JOB svcStyle
    class CHUNK,EMBED,GEN,RERANK,STT,PROC intStyle
    class MILVUS_SEARCH,MILVUS_STORE,MILVUS_CONV,REDIS repoStyle
    class MILVUS_DB,REDIS_DB,GPU infraStyle
```

---

## Project Structure

```
app/
├── main.py                          # FastAPI application factory
├── core/
│   ├── config.py                    # Pydantic Settings (env-driven)
│   ├── gpu.py                       # Shared GPU threading lock
│   └── logging.py                   # Context-aware logging with request IDs
├── api/
│   ├── openai_compat.py             # OpenAI-compatible /v1/* endpoints
│   └── v1/
│       └── endpoints/
│           ├── health.py            # GET /health, GET /ready
│           ├── files.py             # POST /files/{collection}
│           ├── jobs.py              # GET /jobs/{job_id}
│           ├── search.py            # POST /search/{collection}
│           └── conversations.py     # Conversation CRUD + RAG messaging
├── schemas/
│   ├── files.py                     # FileIngestionResponse, FileResult
│   ├── jobs.py                      # JobStatusResponse, FileJobStatus
│   ├── search.py                    # SearchRequest, SearchResult, SearchResponse
│   └── conversations.py            # Conversation & message request/response schemas
├── models/
│   ├── doc.py                       # Document domain model
│   └── conversation.py              # Message, ConversationMeta domain models
├── middleware/
│   ├── auth.py                      # Static API key authentication
│   ├── rate_limit.py                # Sliding-window rate limiter
│   ├── request_context.py           # Request ID injection + duration logging
│   └── errors.py                    # ApiError base class + exception handlers
├── services/
│   ├── public/
│   │   ├── ingest.py                # File ingestion orchestrator
│   │   ├── search.py                # Search dispatcher + reranking
│   │   ├── conversations.py         # RAG conversation orchestrator
│   │   └── job_status.py            # Job polling wrapper
│   └── internal/
│       ├── chunk.py                 # Text splitting + LLM title generation
│       ├── embed.py                 # Google Gemini dense embeddings
│       ├── generate.py              # Google Gemini (Gemma 3) generation (stream + sync)
│       ├── rerank.py                # CrossEncoder reranking with GPU lifecycle
│       ├── speech_to_text.py        # faster-whisper transcription with GPU lifecycle
│       └── process_files.py         # End-to-end file processing pipeline
├── repositories/
│   ├── milvus/
│   │   ├── _client.py               # MilvusClient singleton
│   │   ├── _collection.py           # Collection schema + index creation
│   │   ├── storage.py               # Document upsert / delete
│   │   ├── search.py                # Dense, sparse, hybrid search
│   │   └── conversations.py         # Conversation + message persistence
│   └── redis/
│       ├── _client.py               # Redis client singleton
│       └── job_store.py             # Job lifecycle tracking
└── utils/
    ├── save_upload.py               # File upload persistence
    └── download.py                  # yt-dlp audio downloader

tests/
├── test_search.py                   # 58 tests — search service + endpoints
├── test_conversations.py            # 85 tests — conversation CRUD + RAG
├── test_openai_compat.py            # OpenAI compatibility layer tests
├── test_ingestion.py                # File ingestion pipeline tests
└── test_db.py                       # Milvus repository integration tests

compose.yaml                         # Docker Compose (Milvus, Redis, Open WebUI)
pyproject.toml                       # Project metadata + dependencies (uv)
```

---

## Layer-by-Layer Breakdown

### API Layer

The API layer handles HTTP routing, request deserialization, and response formatting. It contains no business logic.

**REST API (`/api/v1/`)**

| Endpoint                              | Method | Description                                             |
| ------------------------------------- | ------ | ------------------------------------------------------- |
| `/api/v1/health`                      | GET    | Liveness probe                                          |
| `/api/v1/ready`                       | GET    | Readiness probe                                         |
| `/api/v1/files/{collection_name}`     | POST   | Upload files for async ingestion (returns 202 + job ID) |
| `/api/v1/jobs/{job_id}`               | GET    | Poll ingestion job progress                             |
| `/api/v1/search/{collection_name}`    | POST   | Execute vector search (dense/sparse/hybrid)             |
| `/api/v1/conversations`               | POST   | Create a new RAG conversation                           |
| `/api/v1/conversations`               | GET    | List conversations (filterable by collection)           |
| `/api/v1/conversations/{id}`          | GET    | Get conversation with full message history              |
| `/api/v1/conversations/{id}`          | DELETE | Delete conversation and all messages                    |
| `/api/v1/conversations/{id}/messages` | POST   | Send message and receive RAG-augmented response         |

**OpenAI-Compatible API (`/v1/`)**

Allows Open WebUI to use the RAG backend as a standard LLM provider:

| Endpoint                   | Method | Description                                                  |
| -------------------------- | ------ | ------------------------------------------------------------ |
| `/api/v1/models`           | GET    | Lists Milvus collections as models (`rag/{collection_name}`) |
| `/api/v1/chat/completions` | POST   | Full RAG pipeline with OpenAI-format streaming (SSE)         |

### Schema Layer

Pydantic v2 models for request validation and response serialization. Separated from domain models to decouple the API contract from internal representations.

Key schemas:

- **`SearchRequest`** — `query`, `top_k`, `search_type` (dense/sparse/hybrid), `rerank`, `language`
- **`SendMessageRequest`** — `content`, `search_type`, `top_k`, `stream`, `rerank`
- **`SearchResult`** — `doc_id`, `title`, `text`, `score`, `metadata`
- **`SendMessageResponse`** — paired `user_message` + `assistant_message` with sources

### Middleware Layer

Pluggable HTTP middleware for cross-cutting concerns:

| Middleware          | Purpose                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| **Authentication**  | Static API key validation via `Authorization: Bearer` or `X-API-Key` header      |
| **Rate Limiting**   | In-memory sliding-window rate limiter (configurable requests/window per IP+path) |
| **Request Context** | UUID request ID injection, request duration logging                              |
| **Error Handling**  | Structured JSON error responses with `ApiError` hierarchy and request ID tracing |

### Service Layer — Public

Orchestration services that coordinate multiple internal services and repositories to fulfill business operations. Each public service function is `async` and offloads blocking I/O via `asyncio.run_in_executor`.

**Ingestion Service** (`ingest.py`)

- Splits uploads into text files and audio files
- Processes both branches **concurrently** via `asyncio.gather`
- Audio files: GPU-serialized transcription, then processed as text
- Updates Redis job status at each stage (queued → processing → completed/failed)

**Search Service** (`search.py`)

- Dispatches to dense (HNSW/COSINE), sparse (BM25), or hybrid search
- **Overfetch + rerank**: when enabled, fetches `OVERFETCH_MULTIPLIER * top_k` candidates, then applies CrossEncoder reranking to return the best `top_k`

**Conversation Service** (`conversations.py`)

- Full RAG pipeline: **Retrieve → Augment → Generate → Persist**
- Supports both synchronous responses and **Server-Sent Events (SSE) streaming**
- Auto-generates conversation titles from the first user message
- Trims conversation history to a configurable window before sending to the LLM

### Service Layer — Internal

Atomic, single-responsibility services that encapsulate individual ML/AI capabilities:

| Service              | Responsibility                                                   | Provider                |
| -------------------- | ---------------------------------------------------------------- | ----------------------- |
| **Chunking**         | `RecursiveCharacterTextSplitter` with configurable overlap       | LangChain               |
| **Title Generation** | LLM-powered chunk title generation                               | Google Gemma 3          |
| **Embedding**        | Asymmetric dense embeddings (separate document/query task types) | Google Gemini           |
| **Generation**       | RAG answer generation (streaming + non-streaming)                | Google Gemma 3          |
| **Reranking**        | Cross-encoder relevance scoring with GPU lifecycle management    | BAAI/bge-reranker-v2-m3 |
| **Speech-to-Text**   | Batched audio transcription with GPU lifecycle management        | faster-whisper          |
| **File Processing**  | End-to-end pipeline: load → chunk → title → embed → Document     | Composite               |

### Repository Layer

Data access layer that abstracts all persistence concerns. Each repository module exposes pure functions — no business logic.

**Milvus Repositories:**

- **Search** — `dense_search()`, `sparse_search()`, `hybrid_search()` with configurable fusion (Weighted/DBSF ranker or RRF ranker)
- **Storage** — `upsert_documents()`, `delete_documents()` with auto-collection creation
- **Conversations** — Two-collection design (`_conversation_meta`, `_conversation_messages`) with full CRUD
- **Collection Schema** — Dense vector (FLOAT_VECTOR, 768d, HNSW index), sparse vector (SPARSE_FLOAT_VECTOR, BM25 function), plus metadata fields

**Redis Repository:**

- **Job Store** — Hash-based job tracking with per-file granularity at `job:{id}` and `job:{id}:files:{filename}`, with 1-hour TTL auto-expiry

### Core Infrastructure

| Module       | Purpose                                                                          |
| ------------ | -------------------------------------------------------------------------------- |
| `config.py`  | Centralized `pydantic-settings` configuration loaded from `.env` with validation |
| `gpu.py`     | Shared `threading.Lock` preventing GPU OOM between reranker and speech-to-text   |
| `logging.py` | `contextvars`-based request ID propagation with structured logging               |

---

## Key Data Flows

### Document Ingestion Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant API as Files Endpoint
    participant Redis
    participant Ingest as Ingestion Service
    participant STT as Speech-to-Text
    participant Proc as File Processor
    participant Embed as Gemini Embedding
    participant Milvus

    Client->>API: POST /files/{collection} (multipart upload)
    API->>API: Validate file types, save to disk
    API->>Redis: create_job(job_id, filenames)
    API-->>Client: 202 Accepted {job_id}

    Note over API,Ingest: Background Task
    API->>Ingest: ingest_files(job_id, paths, collection)

    par Text Files
        Ingest->>Proc: process_single_file(path)
        Proc->>Proc: Load (PDF/DOCX/TXT/MD)
        Proc->>Proc: Chunk (RecursiveCharacterTextSplitter)
        Proc->>Embed: dense_embed(chunk_texts)
        Embed-->>Proc: float vectors (768d)
        Proc-->>Ingest: Document[]
    and Audio Files
        Ingest->>STT: parse_audio_to_text(audio_paths)
        Note over STT: Acquire GPU lock
        STT->>STT: Whisper transcribe
        Note over STT: Release GPU lock
        STT-->>Ingest: transcript .txt paths
        Ingest->>Proc: process_single_file(transcript)
        Proc-->>Ingest: Document[]
    end

    Ingest->>Milvus: upsert_documents(docs, collection)
    Ingest->>Redis: set_job_result(job_id, count)

    Client->>API: GET /jobs/{job_id}
    API->>Redis: get_job(job_id)
    Redis-->>Client: {status, processed, documents_ingested, ...}
```

### RAG Conversation Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant API as Conversations Endpoint
    participant Conv as Conversation Service
    participant Search as Search Service
    participant Embed as Gemini Embedding
    participant Milvus as Milvus Search
    participant Rerank as CrossEncoder
    participant LLM as Gemma 3
    participant Store as Milvus Conversations

    Client->>API: POST /conversations/{id}/messages
    API->>Conv: send_message(id, content, rerank=true)

    par Load Conversation
        Conv->>Store: get_conversation(id)
        Store-->>Conv: ConversationMeta
    and Load History
        Conv->>Store: get_messages(id)
        Store-->>Conv: Message[]
    end

    Conv->>Search: search_documents(query, collection, rerank=true)
    Search->>Embed: embed_query(query)
    Embed-->>Search: query_vector

    Note over Search: Overfetch: top_k * 2.0
    Search->>Milvus: hybrid_search(vector, text, fetch_k)
    Milvus-->>Search: candidate documents

    Search->>Rerank: rerank(query, candidates)
    Note over Rerank: Acquire GPU lock
    Rerank-->>Search: scored rankings
    Note over Rerank: Release GPU lock

    Search-->>Conv: SearchResult[] (top_k)

    Conv->>Conv: build_context_block(sources)
    Conv->>Conv: build_messages(history + context + query)
    Conv->>LLM: generate(messages)
    LLM-->>Conv: answer

    Conv->>Store: save_messages([user_msg, assistant_msg])
    Conv-->>Client: {user_message, assistant_message, sources}
```

### Search Pipeline with Reranking

```mermaid
flowchart LR
    Q[User Query] --> EMB[Embed Query<br/><i>Gemini RETRIEVAL_QUERY</i>]

    EMB --> D{Search Type}
    Q --> D

    D -->|dense| DENSE[HNSW / COSINE<br/><i>Dense Vector Search</i>]
    D -->|sparse| SPARSE[BM25<br/><i>Sparse Text Search</i>]
    D -->|hybrid| HYBRID[WeightedRanker / RRF<br/><i>Dense + Sparse Fusion</i>]

    DENSE --> OF{Rerank<br/>Enabled?}
    SPARSE --> OF
    HYBRID --> OF

    OF -->|No| RET1[Retrieve top_k results]
    RET1 --> RES[Return top_k results]
    OF -->|Yes| OV[Overfetch<br/><i>fetch_k = top_k * 2.0</i>]
    OV --> RET2[Retrieve fetch_k results]
    RET2 --> CE[CrossEncoder Rerank<br/><i>BAAI/bge-reranker-v2-m3</i>]
    CE --> TRIM[Trim to top_k]
    TRIM --> RES
```

---

## Concurrency and GPU Management

### Async Architecture

All API endpoints are `async`. Every blocking operation (Milvus queries, embedding API calls, LLM generation, file I/O) is offloaded to the default thread-pool executor via `asyncio.run_in_executor`, ensuring the event loop remains responsive under concurrent load.

```python
# Pattern used throughout the codebase:
async def search_documents(query, collection_name, ...):
    loop = asyncio.get_running_loop()
    query_vector = await embed_query(query)                          # async
    hits = await loop.run_in_executor(None, _run_hybrid_search, ...) # offloaded
    rankings = await loop.run_in_executor(None, _rerank_sync, ...)   # offloaded
```

### GPU Memory Safety

The system is designed to run on machines with limited GPU VRAM (4 GB). Two services require exclusive GPU access:

1. **Reranker** (CrossEncoder) — loads model to CUDA, reranks, moves back to CPU, frees VRAM
2. **Speech-to-Text** (faster-whisper) — loads CTranslate2 weights to CUDA, transcribes, unloads

A single `threading.Lock` in `app/core/gpu.py` serializes GPU access:

```mermaid
sequenceDiagram
    participant R as Reranker
    participant L as GPU Lock
    participant S as Speech-to-Text

    R->>L: acquire()
    Note over R: model.to("cuda")
    Note over R: rerank(...)
    Note over R: model.to("cpu")
    Note over R: torch.cuda.empty_cache()
    R->>L: release()

    S->>L: acquire()
    Note over S: ct2_model.load_model()
    Note over S: transcribe(...)
    Note over S: ct2_model.unload_model()
    Note over S: torch.cuda.empty_cache()
    S->>L: release()
```

---

## Technology Stack

| Category              | Technology                               | Purpose                                                |
| --------------------- | ---------------------------------------- | ------------------------------------------------------ |
| **Framework**         | FastAPI + Uvicorn                        | Async HTTP server with OpenAPI docs                    |
| **Validation**        | Pydantic v2                              | Request/response schemas and domain models             |
| **Vector Database**   | Milvus 2.6                               | Dense (HNSW) + sparse (BM25) vector storage and search |
| **Cache / Job Store** | Redis 8                                  | Async job tracking with TTL-based expiry               |
| **Embeddings**        | Google Gemini (`gemini-embedding-001`)   | 768-dimensional asymmetric embeddings                  |
| **LLM**               | Google Gemma 3 (`gemma-3-27b-it`)        | RAG answer generation and title generation             |
| **Reranking**         | `BAAI/bge-reranker-v2-m3` (CrossEncoder) | Cross-encoder relevance scoring                        |
| **Speech-to-Text**    | faster-whisper (CTranslate2)             | Batched audio transcription                            |
| **Text Processing**   | LangChain                                | Document loaders (PDF, DOCX, TXT) and text splitting   |
| **Streaming**         | SSE (sse-starlette)                      | Token-by-token response streaming                      |
| **UI**                | Open WebUI v0.8                          | Chat interface via OpenAI-compatible API               |
| **Package Manager**   | uv                                       | Fast Python dependency management                      |
| **Containerization**  | Docker Compose                           | Milvus + etcd + MinIO + Redis + Open WebUI             |

---

## Infrastructure & Deployment

```mermaid
graph LR
    subgraph Host Machine
        APP[FastAPI Backend<br/><i>:8000</i>]
    end

    subgraph Docker Compose
        ETCD[etcd<br/><i>Milvus metadata</i>]
        MINIO[MinIO<br/><i>Object storage</i>]
        MILVUS[Milvus Standalone<br/><i>:19530</i>]
        REDIS[Redis<br/><i>:6379</i>]
        WEBUI[Open WebUI<br/><i>:8080</i>]
    end

    WEBUI -->|OpenAI API| APP
    APP -->|gRPC| MILVUS
    APP -->|TCP| REDIS
    MILVUS --> ETCD
    MILVUS --> MINIO

    classDef docker fill:#2496ED,stroke:#1A6DB5,color:#fff
    class ETCD,MINIO,MILVUS,REDIS,WEBUI docker
```

The `compose.yaml` provisions the full infrastructure stack:

- **Milvus Standalone** with etcd (metadata) and MinIO (object storage) backends
- **Redis** with AOF persistence and password authentication
- **Open WebUI** configured to use the FastAPI backend as its OpenAI provider

---

## Configuration Reference

All settings are managed via environment variables (`.env` file), loaded through `pydantic-settings`:

| Variable                      | Default                       | Description                                        |
| ----------------------------- | ----------------------------- | -------------------------------------------------- |
| `GOOGLE_API_KEY`              | —                             | Google API key for Embeddings + Generation         |
| `MAX_TOKENS`                  | `1024`                        | Maximum chunk size (characters)                    |
| `OVERLAP_TOKENS`              | `200`                         | Overlap between consecutive chunks                 |
| `EMBEDDING_MODEL`             | `models/gemini-embedding-001` | Gemini embedding model                             |
| `EMBEDDING_DIM`               | `768`                         | Embedding vector dimensionality                    |
| `FUSION_METHOD`               | `weighted`                    | Hybrid search fusion: `weighted`, `dbsf`, or `rrf` |
| `FUSION_ALPHA`                | `0.7`                         | Dense vs sparse weight (1.0 = all dense)           |
| `RERANKER_MODEL`              | `BAAI/bge-reranker-v2-m3`     | CrossEncoder model for reranking                   |
| `OVERFETCH_MULTIPLIER`        | `2.0`                         | Overfetch factor before reranking                  |
| `GENERATION_MODEL`            | `gemma-3-27b-it`              | LLM model for RAG generation                       |
| `GENERATION_SEARCH_TYPE`      | `hybrid`                      | Default search type for RAG                        |
| `GENERATION_RAG_TOP_K`        | `5`                           | Documents retrieved per query                      |
| `GENERATION_HISTORY_TURNS`    | `10`                          | Max conversation turns sent to LLM                 |
| `MILVUS_URI`                  | `http://localhost:19530`      | Milvus connection URI                              |
| `REDIS_HOST`                  | `localhost`                   | Redis host                                         |
| `REDIS_PORT`                  | `6379`                        | Redis port                                         |
| `OPENWEBUI_RERANKING_ENABLED` | `True`                        | Enable reranking for Open WebUI queries            |
