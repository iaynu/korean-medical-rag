"""서로 다른 검색기의 결과를 하나의 순위로 합칩니다.

원본 노트북의 ``merge_candidates`` 는 이름과 달리 융합을 하지 않았습니다.
BM25 결과를 딕셔너리에 넣고, 벡터 결과 중 없는 것만 추가한 뒤, 삽입 순서대로
잘라서 반환했습니다. 두 검색기가 매긴 순위는 어디에도 반영되지 않았습니다.

    # 원본
    merged = {}
    for it in bm_items:
        merged[get_id(...)] = it
    for it in vec_items:
        if key not in merged:
            merged[key] = it
    return list(merged.values())[:max_items]

두 검색기 모두가 1위로 꼽은 문서와 한쪽에서만 20위로 걸린 문서가 똑같은
자격으로 재랭커에 넘어갔습니다. 재랭커가 뒤에서 정리해 주긴 하지만,
CrossEncoder는 후보 하나당 한 번씩 forward pass를 돌리므로 후보 수에 비례해
느려집니다. 순위가 있으면 후보를 안전하게 좁힐 수 있습니다 —
:attr:`~korean_medical_rag.config.RagConfig.fusion_top_n` 이 그 용도입니다.

다만 원본이 실제로 쓴 ``hybrid_rerank`` 경로에서는 후보 40건이 ``max_items=40``
에 걸리지 않아 잘림이 없었고, 재랭커가 40건을 전부 다시 채점했습니다. 즉 이
결함이 최종 지표를 크게 망쳤다고 보기는 어렵습니다. 순서가 실제로 결과를
바꾸는 곳은 재랭커를 건너뛰는 ``mode="hybrid"`` (``merged[:5]``) 경로입니다.

**왜 점수를 그냥 더하지 않았나** — 두 점수는 단위가 다릅니다. 벡터 검색은
거리를 음수로 뒤집은 값(대략 -2 ~ 0), BM25는 상한이 없는 값(0 ~ 50+)입니다.
그대로 더하면 BM25가 언제나 이깁니다. min-max 정규화도 방법이지만, 질의마다
점수 분포가 달라서 배치 전체에 일관되게 적용하기 어렵습니다.

**RRF (Reciprocal Rank Fusion)** 는 점수를 아예 보지 않고 순위만 씁니다.

    score(d) = Σ_r  1 / (k + rank_r(d))

점수 스케일에 영향받지 않고, 정규화 파라미터를 튜닝할 필요가 없으며,
검색기를 더 붙여도 그대로 동작합니다. k는 상위권 가중치를 얼마나 완만하게
할지 정하는 값으로, 원 논문(Cormack et al., 2009)의 60을 기본값으로 씁니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "RetrievedDoc",
    "FusedDoc",
    "reciprocal_rank_fusion",
    "fusion_summary",
    "detect_id_collisions",
    "DEFAULT_RRF_K",
]

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RetrievedDoc:
    """검색기가 반환한 문서 하나.

    ``score`` 는 검색기마다 단위가 다릅니다. 순위 비교에는 쓰지 않고
    디버깅과 로깅 용도로만 보관합니다.

    >>> d = RetrievedDoc(doc_id="a", text="당뇨병 개요")
    >>> d.doc_id
    'a'
    >>> d.score is None
    True
    """

    doc_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id는 비어 있을 수 없습니다")


@dataclass(frozen=True)
class FusedDoc:
    """융합 후 문서. 어느 검색기가 몇 위로 올렸는지 함께 남깁니다.

    ``contributions`` 는 "이 문서가 왜 최종 후보에 올라왔는가"를 설명합니다.
    벡터에서만 올라온 문서가 정답이었다면 BM25 쪽을, 반대라면 임베딩 쪽을
    의심할 근거가 됩니다. 원본 파이프라인에는 이 정보가 없어서 검색 실패의
    원인을 사후에 따져볼 수 없었습니다.
    """

    doc: RetrievedDoc
    rrf_score: float
    contributions: Mapping[str, int]

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id

    @property
    def text(self) -> str:
        return self.doc.text


def detect_id_collisions(
    ranked_lists: Mapping[str, Sequence[RetrievedDoc]],
) -> Dict[str, int]:
    """같은 ID로 서로 다른 본문이 들어온 경우를 찾습니다.

    :class:`BM25Retriever` 는 인덱싱 시점에 충돌을 잡지만, 벡터 검색기는
    Chroma가 돌려준 메타데이터를 그대로 쓰므로 그 시점이 없습니다. 융합은
    ID로 문서를 합치기 때문에, 여기가 서로 다른 문서가 하나로 뭉개지는
    마지막 지점입니다. 조용히 넘어가는 대신 세어서 보고합니다.

    Returns:
        충돌한 doc_id → 그 ID로 관측된 서로 다른 본문의 수.
        충돌이 없으면 빈 딕셔너리.

    >>> a = RetrievedDoc("dup", "본문 하나")
    >>> b = RetrievedDoc("dup", "본문 둘")
    >>> detect_id_collisions({"v": [a], "b": [b]})
    {'dup': 2}

    같은 ID에 같은 본문이면 충돌이 아닙니다. 두 검색기가 같은 문서를
    찾아낸 정상적인 경우입니다.

    >>> detect_id_collisions({"v": [a], "b": [RetrievedDoc("dup", "본문 하나")]})
    {}
    """
    texts_by_id: Dict[str, set[str]] = {}
    for ranked in ranked_lists.values():
        for doc in ranked:
            texts_by_id.setdefault(doc.doc_id, set()).add(doc.text)
    return {doc_id: len(texts) for doc_id, texts in texts_by_id.items() if len(texts) > 1}


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[RetrievedDoc]],
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> List[FusedDoc]:
    """여러 검색 결과를 RRF로 융합합니다.

    Args:
        ranked_lists: 검색기 이름 → 순위대로 정렬된 문서 리스트.
        k: RRF 상수. 클수록 상위권과 하위권의 점수 차가 완만해집니다.
        top_n: 반환할 상위 문서 수. None이면 전부.

    Returns:
        RRF 점수 내림차순 문서 리스트. 동점이면 doc_id 오름차순으로
        정렬해 실행할 때마다 결과가 달라지지 않게 합니다.

    Raises:
        ValueError: k가 1 미만이거나 top_n이 1 미만일 때.

    핵심 성질은 **합의가 단독 1위를 이긴다**는 것입니다. 아래에서 A는
    벡터 검색만의 1위, B는 BM25만의 1위, C는 양쪽 모두의 2위입니다.

    >>> vec = [RetrievedDoc("A", "a"), RetrievedDoc("C", "c"), RetrievedDoc("D", "d")]
    >>> bm  = [RetrievedDoc("B", "b"), RetrievedDoc("C", "c"), RetrievedDoc("D", "d")]
    >>> fused = reciprocal_rank_fusion({"vector": vec, "bm25": bm}, k=1)
    >>> [f.doc_id for f in fused]
    ['C', 'A', 'B', 'D']

    C가 1위입니다. 두 검색기가 독립적으로 상위에 올렸다는 사실이,
    한쪽에서만 1위인 것보다 강한 신호이기 때문입니다.

    >>> round(fused[0].rrf_score, 4) == round(1/3 + 1/3, 4)
    True
    >>> dict(fused[0].contributions)
    {'vector': 2, 'bm25': 2}

    한쪽에서만 나온 문서도 버리지 않습니다. 순위만 뒤로 밀립니다.

    >>> [f.doc_id for f in fused if len(f.contributions) == 1]
    ['A', 'B']

    점수 스케일은 결과에 영향을 주지 않습니다. BM25 점수를 1000배로
    키워도 순위가 동일합니다. 원본처럼 점수를 직접 더했다면 BM25가
    모든 순위를 지배했을 상황입니다.

    >>> big = [RetrievedDoc(d.doc_id, d.text, score=9999.0) for d in bm]
    >>> [f.doc_id for f in reciprocal_rank_fusion({"vector": vec, "bm25": big}, k=1)]
    ['C', 'A', 'B', 'D']

    같은 검색기가 같은 문서를 중복 반환해도 점수가 부풀지 않습니다.

    >>> dup = [RetrievedDoc("A", "a"), RetrievedDoc("A", "a"), RetrievedDoc("B", "b")]
    >>> [(f.doc_id, dict(f.contributions)) for f in reciprocal_rank_fusion({"v": dup}, k=1)]
    [('A', {'v': 1}), ('B', {'v': 3})]

    빈 입력은 빈 결과입니다. 예외를 던지지 않습니다. 배치 평가 중 질의
    하나가 결과 없음이라고 전체가 멈추면 곤란하기 때문입니다.

    >>> reciprocal_rank_fusion({})
    []
    >>> reciprocal_rank_fusion({"vector": [], "bm25": []})
    []
    """
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다 (받은 값: {k})")
    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n은 1 이상이어야 합니다 (받은 값: {top_n})")

    scores: Dict[str, float] = {}
    docs: Dict[str, RetrievedDoc] = {}
    contributions: Dict[str, Dict[str, int]] = {}

    for retriever_name, ranked in ranked_lists.items():
        seen_in_this_list: set[str] = set()
        for rank, doc in enumerate(ranked, start=1):
            # 한 검색기가 같은 문서를 두 번 반환하면 첫 순위만 인정합니다.
            # 중복을 그대로 더하면 그 문서만 부당하게 높아집니다.
            if doc.doc_id in seen_in_this_list:
                continue
            seen_in_this_list.add(doc.doc_id)

            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc.doc_id, doc)
            contributions.setdefault(doc.doc_id, {})[retriever_name] = rank

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        ordered = ordered[:top_n]

    return [
        FusedDoc(
            doc=docs[doc_id],
            rrf_score=score,
            contributions=dict(contributions[doc_id]),
        )
        for doc_id, score in ordered
    ]


def fusion_summary(fused: Sequence[FusedDoc]) -> str:
    """융합 결과를 한 줄로 요약합니다. 로그용.

    >>> vec = [RetrievedDoc("A", "a"), RetrievedDoc("B", "b")]
    >>> bm = [RetrievedDoc("B", "b"), RetrievedDoc("C", "c")]
    >>> print(fusion_summary(reciprocal_rank_fusion({"vector": vec, "bm25": bm})))
    후보 3건 (양쪽 1건, vector 단독 1건, bm25 단독 1건)
    """
    if not fused:
        return "후보 0건"

    both = sum(1 for f in fused if len(f.contributions) > 1)
    only: Dict[str, int] = {}
    for f in fused:
        if len(f.contributions) == 1:
            name = next(iter(f.contributions))
            only[name] = only.get(name, 0) + 1

    parts = [f"양쪽 {both}건"] + [f"{n} 단독 {c}건" for n, c in sorted(only.items(), reverse=True)]
    return f"후보 {len(fused)}건 ({', '.join(parts)})"
