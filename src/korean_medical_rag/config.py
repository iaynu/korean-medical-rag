"""파이프라인 설정.

원본 노트북은 경로·k값·모델명이 셀 여기저기에 흩어져 있었습니다.
실험을 재현하려면 어떤 셀을 어떤 값으로 돌렸는지 기억해야 했습니다.
설정을 한 곳에 모으고 생성 시점에 검증합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from .exceptions import ConfigError

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_LLM_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"


@dataclass(frozen=True)
class RagConfig:
    """검색·재랭킹·생성 전 구간의 설정.

    Attributes:
        db_path: Chroma 영속 디렉터리 경로.
        collection_name: Chroma 컬렉션 이름.
        embed_model: 문서/질의 임베딩 모델.
        rerank_model: CrossEncoder 재랭킹 모델.
        llm_model: 답변 생성 LLM.
        vector_k: 벡터 검색이 가져올 후보 수.
        bm25_k: BM25 검색이 가져올 후보 수.
        rerank_k: 재랭킹 후 LLM에 넘길 문서 수.
        rrf_k: RRF 순위 상수. 클수록 상위권 가중치가 완만해집니다.
        fusion_top_n: 재랭커에 넘길 후보 수 상한. None이면 전부 넘깁니다.
            CrossEncoder는 후보 하나당 forward pass를 한 번씩 돌리므로
            후보 수에 비례해 느려집니다. RRF로 순위를 매긴 뒤 상위 일부만
            넘기면 그만큼 비용이 줄어듭니다.
        max_new_tokens: 생성 토큰 상한.
        tokenizer_mode: BM25용 한국어 토크나이저 모드.

    >>> RagConfig(db_path="/tmp/db").rerank_k
    5
    >>> RagConfig(db_path="/tmp/db", vector_k=0)
    Traceback (most recent call last):
        ...
    korean_medical_rag.exceptions.ConfigError: vector_k는 1 이상이어야 합니다 (받은 값: 0)
    """

    db_path: str
    collection_name: str = "med_knowledge"

    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_model: str = DEFAULT_RERANK_MODEL
    llm_model: str = DEFAULT_LLM_MODEL

    vector_k: int = 20
    bm25_k: int = 20
    rerank_k: int = 5
    rrf_k: int = 60
    fusion_top_n: int | None = None

    max_new_tokens: int = 512
    tokenizer_mode: str = "auto"

    def __post_init__(self) -> None:
        if not self.db_path:
            raise ConfigError("db_path는 비어 있을 수 없습니다")

        for name in ("vector_k", "bm25_k", "rerank_k", "max_new_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"{name}은 정수여야 합니다 (받은 값: {value!r})")
            if value < 1:
                raise ConfigError(f"{name}는 1 이상이어야 합니다 (받은 값: {value})")

        if not isinstance(self.rrf_k, int) or self.rrf_k < 1:
            raise ConfigError(f"rrf_k는 1 이상의 정수여야 합니다 (받은 값: {self.rrf_k!r})")

        # 재랭킹이 후보보다 많은 문서를 뽑을 수는 없습니다. 원본에서는
        # 이 조합을 실수로 만들어도 조용히 후보 수만큼만 반환됐습니다.
        candidates = self.vector_k + self.bm25_k
        if self.rerank_k > candidates:
            raise ConfigError(
                f"rerank_k({self.rerank_k})가 후보 총합({candidates})보다 큽니다"
            )

        if self.fusion_top_n is not None:
            if not isinstance(self.fusion_top_n, int) or isinstance(self.fusion_top_n, bool):
                raise ConfigError(
                    f"fusion_top_n은 정수 또는 None이어야 합니다 (받은 값: {self.fusion_top_n!r})"
                )
            if self.fusion_top_n < self.rerank_k:
                raise ConfigError(
                    f"fusion_top_n({self.fusion_top_n})이 rerank_k({self.rerank_k})보다 작습니다. "
                    f"재랭커에 넘긴 후보보다 많이 뽑을 수 없습니다."
                )

        if self.tokenizer_mode not in ("auto", "morph", "ngram", "whitespace"):
            raise ConfigError(
                f"tokenizer_mode는 auto/morph/ngram/whitespace 중 하나여야 합니다 "
                f"(받은 값: {self.tokenizer_mode!r})"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RagConfig":
        """딕셔너리에서 설정을 만듭니다. 모르는 키는 거부합니다.

        오타 난 키를 조용히 무시하면 기본값으로 실험이 돌아가고,
        결과가 왜 안 바뀌는지 한참 헤매게 됩니다.

        >>> RagConfig.from_mapping({"db_path": "/tmp/db", "rerank_k": 3}).rerank_k
        3
        >>> RagConfig.from_mapping({"db_path": "/tmp/db", "rerank_kk": 3})
        Traceback (most recent call last):
            ...
        korean_medical_rag.exceptions.ConfigError: 알 수 없는 설정 키: rerank_kk
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"알 수 없는 설정 키: {', '.join(unknown)}")
        return cls(**dict(data))
