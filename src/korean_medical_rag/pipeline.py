"""검색 → 융합 → 재랭킹 → 생성 오케스트레이션.

각 단계는 주입받은 객체에 위임합니다. 파이프라인 자체는 torch도
chromadb도 import하지 않으므로, 가짜 검색기·재랭커·생성기를 넣으면
전체 흐름을 CPU에서 테스트할 수 있습니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Sequence

from .config import RagConfig
from .fusion import (
    FusedDoc,
    RetrievedDoc,
    detect_id_collisions,
    fusion_summary,
    reciprocal_rank_fusion,
)
from .prompts import QuestionType, build_prompt
from .retrieval import Retriever

__all__ = ["RagResult", "RagPipeline", "Reranker", "Generator"]


class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[FusedDoc], top_k: int) -> List[FusedDoc]:
        ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class RagResult:
    """한 질의의 처리 결과와 중간 산출물.

    중간 결과를 모두 남기는 이유는 답변이 틀렸을 때 어느 단계가 문제였는지
    가려내기 위해서입니다. 정답 문서가 ``fused`` 에는 있는데 ``reranked``
    에서 빠졌다면 재랭커를, ``fused`` 에도 없다면 검색 단계를 봐야 합니다.
    원본 파이프라인은 최종 답변만 반환해서 이 구분이 불가능했습니다.
    """

    question: str
    q_type: int
    answer: str
    fused: List[FusedDoc]
    reranked: List[FusedDoc]
    prompt: str
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    id_collisions: Mapping[str, int] = field(default_factory=dict)

    @property
    def retrieved_ids(self) -> List[str]:
        return [d.doc_id for d in self.reranked]

    def trace(self) -> str:
        """단계별 요약을 여러 줄 문자열로 만듭니다.

        >>> r = RagResult("q", 1, "a", [], [], "p", {"retrieve": 12.3})
        >>> print(r.trace())
        질문: q (유형 1)
        융합: 후보 0건
        재랭킹: 0건 선정
        소요: retrieve 12.3ms

        ID 충돌이 있으면 경고를 덧붙입니다. 조용히 넘어가면 서로 다른
        문서가 하나로 뭉개진 것을 알 수 없습니다.

        >>> r = RagResult("q", 1, "a", [], [], "p", id_collisions={"dup": 2})
        >>> print(r.trace())
        질문: q (유형 1)
        융합: 후보 0건
        재랭킹: 0건 선정
        경고: ID 충돌 1건 — 서로 다른 본문이 같은 ID로 합쳐졌습니다 (dup)
        """
        lines = [
            f"질문: {self.question} (유형 {self.q_type})",
            f"융합: {fusion_summary(self.fused)}",
            f"재랭킹: {len(self.reranked)}건 선정",
        ]
        if self.id_collisions:
            sample = ", ".join(sorted(self.id_collisions)[:3])
            lines.append(
                f"경고: ID 충돌 {len(self.id_collisions)}건 — "
                f"서로 다른 본문이 같은 ID로 합쳐졌습니다 ({sample})"
            )
        if self.timings_ms:
            timing = " ".join(f"{k} {v:.1f}ms" for k, v in self.timings_ms.items())
            lines.append(f"소요: {timing}")
        return "\n".join(lines)


class RagPipeline:
    """하이브리드 RAG 파이프라인.

    Args:
        config: 설정.
        retrievers: 검색기들. 이름은 각 객체의 ``name`` 속성을 씁니다.
        reranker: 재랭커. None이면 재랭킹 없이 RRF 상위를 그대로 씁니다.
        generator: 생성기. None이면 프롬프트만 만들고 답변은 빈 문자열.
        tokenizer: chat template용. None이면 평문 프롬프트.
    """

    def __init__(
        self,
        config: RagConfig,
        retrievers: Sequence[Retriever],
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        tokenizer: Any = None,
    ) -> None:
        if not retrievers:
            raise ValueError("검색기가 최소 하나는 필요합니다")

        names = [r.name for r in retrievers]
        if len(set(names)) != len(names):
            raise ValueError(f"검색기 이름이 중복됩니다: {names}")

        self.config = config
        self.retrievers = list(retrievers)
        self.reranker = reranker
        self.generator = generator
        self.tokenizer = tokenizer

    def _k_for(self, retriever: Retriever) -> int:
        if retriever.name == "bm25":
            return self.config.bm25_k
        return self.config.vector_k

    def answer(self, question: str, q_type: int | QuestionType) -> RagResult:
        """질의 하나를 끝까지 처리합니다."""
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        ranked_lists: Dict[str, List[RetrievedDoc]] = {}
        for retriever in self.retrievers:
            ranked_lists[retriever.name] = retriever.retrieve(
                question, self._k_for(retriever)
            )
        timings["retrieve"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        collisions = detect_id_collisions(ranked_lists)
        # fusion_top_n으로 재랭커에 넘길 후보를 미리 좁힙니다. CrossEncoder는
        # 후보당 forward pass를 한 번씩 돌리므로 여기서 자른 만큼 그대로
        # 비용이 줄어듭니다. None이면 원본과 동일하게 전부 넘깁니다.
        fused = reciprocal_rank_fusion(
            ranked_lists, k=self.config.rrf_k, top_n=self.config.fusion_top_n
        )
        timings["fuse"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.reranker is not None and fused:
            reranked = self.reranker.rerank(question, fused, self.config.rerank_k)
        else:
            reranked = fused[: self.config.rerank_k]
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        prompt = build_prompt(self.tokenizer, question, q_type, reranked)

        t0 = time.perf_counter()
        answer = self.generator.generate(prompt) if self.generator is not None else ""
        timings["generate"] = (time.perf_counter() - t0) * 1000

        return RagResult(
            question=question,
            q_type=int(q_type),
            answer=answer,
            fused=fused,
            reranked=list(reranked),
            prompt=prompt,
            timings_ms=timings,
            id_collisions=collisions,
        )
