"""테스트 공용 픽스처와 가짜 객체.

가짜 검색기·재랭커·생성기를 두는 이유는 파이프라인 로직을 모델 없이
검증하기 위해서입니다. 실제 모델을 쓰면 테스트가 느려지고, GPU가 없는
환경에서는 아예 돌지 않습니다.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest

from korean_medical_rag.fusion import FusedDoc, RetrievedDoc


@pytest.fixture
def medical_corpus() -> List[str]:
    """조사가 붙은 한국어 문장들. 토크나이저 차이가 드러나도록 구성했습니다."""
    return [
        "당뇨병의 주요 합병증은 신장 질환과 망막병증이다.",
        "고혈압 환자는 저염식을 유지해야 한다.",
        "인슐린은 췌장의 베타세포에서 분비되는 호르몬이다.",
        "천식 발작 시에는 기관지 확장제를 사용한다.",
        "당뇨병 환자에게 규칙적인 운동이 권장된다.",
    ]


@pytest.fixture
def corpus_metas(medical_corpus) -> List[dict]:
    return [{"file_id": f"med_{i:03d}"} for i in range(len(medical_corpus))]


class FakeRetriever:
    """미리 정해둔 문서를 순서대로 반환하는 검색기."""

    def __init__(self, name: str, docs: Sequence[RetrievedDoc]) -> None:
        self.name = name
        self._docs = list(docs)
        self.calls: List[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> List[RetrievedDoc]:
        self.calls.append((query, k))
        return self._docs[:k]


class FakeReranker:
    """문서 순서를 뒤집어서 반환합니다.

    재랭킹이 실제로 순서를 바꿨는지 확인하려면, 입력과 다른 순서를
    내놓는 것이 확실합니다.
    """

    def __init__(self) -> None:
        self.calls: List[tuple[str, int]] = []

    def rerank(self, query: str, docs: Sequence[FusedDoc], top_k: int) -> List[FusedDoc]:
        self.calls.append((query, top_k))
        return list(reversed(list(docs)))[:top_k]


class FakeGenerator:
    """프롬프트를 기록하고 고정 답변을 반환합니다."""

    def __init__(self, answer: str = "테스트 답변") -> None:
        self.answer = answer
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture
def fake_retriever_factory():
    return FakeRetriever


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    return FakeGenerator()


def docs(*ids: str) -> List[RetrievedDoc]:
    """ID만으로 간단히 문서 리스트를 만듭니다."""
    return [RetrievedDoc(doc_id=i, text=f"본문 {i}", metadata={"file_id": i}) for i in ids]
