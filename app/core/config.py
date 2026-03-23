import random
from pathlib import Path
from typing import Literal, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    DEBUG_MODE_ENABLED: bool = False
    OPENWEBUI_RERANKING_ENABLED: bool = True

    # llm provider api key
    GOOGLE_API_KEY: Optional[str] = None

    # allowed file types for upload
    ALLOWED_TEXT_EXTS: tuple[str, ...] = (".pdf", ".txt", ".docx", ".doc", ".md")
    ALLOWED_AUDIO_EXTS: tuple[str, ...] = (".mp3", ".wav", ".ogg", ".flac", ".aac")

    # chunking config
    MAX_TOKENS: int = 1024
    OVERLAP_TOKENS: int = 200

    # title generation
    TITLE_GEN_ENABLED: bool = False
    TITLE_GEN_MODEL: str = "gemma-3-27b-it"
    TITLE_GEN_TEMPERATURE: float = 0.3
    TITLE_MAX_TOKENS: int = 50

    # Speech to text
    SPEECH_TO_TEXT_MODEL_SIZE: str = "medium"

    # embedding
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 768  # must match the actual dimension of the embedding model

    # hybrid search fusion parameters
    FUSION_METHOD: Literal["weighted", "dbsf", "rrf"] = "weighted"
    RRF_K: int = 2
    FUSION_ALPHA: float = 0.7

    # reranking
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    OVERFETCH_MULTIPLIER: float = 2.0  # scales top_k by this factor before reranking

    # generation (RAG chat)
    GENERATION_MODEL: str = "gemma-3-27b-it"
    GENERATION_MAX_TOKENS: int = 2048
    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_HISTORY_TURNS: int = 10  # max conversation turns sent to LLM
    GENERATION_RAG_TOP_K: int = 5  # docs to retrieve per query
    GENERATION_SEARCH_TYPE: Literal["dense", "sparse", "hybrid"] = "hybrid"

    # milvus connection
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_DB_NAME: str = "default"
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_TOKEN: str = ""
    MILVUS_TIMEOUT_SEC: float = 30.0

    # milvus index / search
    MILVUS_METRIC_TYPE: str = "COSINE"
    MILVUS_INDEX_TYPE: str = "HNSW"
    MILVUS_HNSW_M: int = 16
    MILVUS_HNSW_EF_CONSTRUCTION: int = 200
    MILVUS_HNSW_EF: int = 64

    # milvus BM25
    MILVUS_BM25_K1: float = 1.5
    MILVUS_BM25_B: float = 0.75

    # milvus collection
    MILVUS_INSERT_BATCH_SIZE: int = 512
    MILVUS_ENABLE_FULLTEXT: bool = False
    MILVUS_TEXT_MAX_LENGTH: int = 9000

    # conversation storage
    CONVERSATION_META_COLLECTION: str = "_conversation_meta"
    CONVERSATION_MSG_COLLECTION: str = "_conversation_messages"

    # redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "dev-redis-password"
    REDIS_DB: int = 0
    REDIS_JOB_TTL_SEC: int = 3600  # 1 hours

    # local storage
    LOCAL_STORAGE_PATH: str = "./.storage"

    @field_validator("FUSION_ALPHA")
    @classmethod
    def fusion_alpha_must_be_between_0_and_1(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("FUSION_ALPHA must be between 0 and 1.")
        return v

    @field_validator("RRF_K")
    @classmethod
    def rrf_k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RRF_K must be a positive integer.")
        return v

    # Derived storage paths (computed, not read from env)
    @property
    def AUDIO_STORAGE_PATH(self) -> Path:
        return Path(self.LOCAL_STORAGE_PATH) / "audio_files"

    @property
    def TRANSCRIPT_STORAGE_PATH(self) -> Path:
        return Path(self.LOCAL_STORAGE_PATH) / "transcripts"

    @property
    def CHUNK_STORAGE_PATH(self) -> Path:
        return Path(self.LOCAL_STORAGE_PATH) / "chunks"


settings = Settings()

# Shared RNG (seeded for reproducibility)
rng = random.Random(42)
