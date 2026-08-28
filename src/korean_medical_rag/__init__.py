"""한국어 의료 QA를 위한 하이브리드 RAG 파이프라인.

벡터 검색과 BM25를 RRF로 융합하고, CrossEncoder로 재랭킹한 뒤
LLM으로 답변을 생성합니다.
"""

from __future__ import annotations

from .config import RagConfig
from .exceptions import (
    ConfigError,
    CorpusError,
    DuplicateDocumentIdError,
    EmptyCorpusError,
    GenerationError,
    RagError,
    RetrievalError,
)
from .fusion import RetrievedDoc, reciprocal_rank_fusion
from .prompts import QuestionType, build_prompt
from .retrieval import BM25Retriever
from .tokenization import get_tokenizer

__version__ = "0.1.0"

__all__ = [
    "RagConfig",
    "RagError",
    "ConfigError",
    "CorpusError",
    "EmptyCorpusError",
    "DuplicateDocumentIdError",
    "RetrievalError",
    "GenerationError",
    "RetrievedDoc",
    "reciprocal_rank_fusion",
    "QuestionType",
    "build_prompt",
    "BM25Retriever",
    "get_tokenizer",
    "__version__",
]
