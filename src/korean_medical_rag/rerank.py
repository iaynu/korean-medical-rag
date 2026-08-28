"""CrossEncoder 재랭킹.

``sentence_transformers`` import를 이 모듈 안에, 그것도 메서드 안에
가둡니다. 파이프라인·융합·프롬프트 로직을 torch 없이 테스트하기 위해서입니다.
이 프로젝트에서 실제로 시간을 잡아먹은 문제 대부분은 모델이 아니라 검색
로직과 데이터 쪽이었고, 그걸 확인하려고 매번 7.8B 모델을 올릴 이유는 없습니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

from .config import DEFAULT_RERANK_MODEL
from .fusion import FusedDoc

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

__all__ = ["CrossEncoderReranker", "resolve_device"]


def resolve_device(preferred: str | None = None) -> str:
    """사용할 디바이스를 정합니다.

    원본은 ``"cuda" if torch.cuda.is_available() else "cpu"`` 였습니다.
    Apple Silicon에서는 MPS가 있어도 CPU로 떨어졌습니다.
    """
    if preferred:
        return preferred

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    """질의-문서 쌍을 CrossEncoder로 점수 매겨 상위 k건을 고릅니다.

    모델은 첫 :meth:`rerank` 호출 때 올립니다. 파이프라인을 조립만 하고
    실행하지 않는 경우(설정 검증, 프롬프트 확인)에 모델을 받지 않게 하려는
    의도입니다.
    """

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size는 1 이상이어야 합니다 (받은 값: {batch_size})")
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model: "CrossEncoder | None" = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device()
        return self._device

    def _load(self) -> "CrossEncoder":
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, docs: Sequence[FusedDoc], top_k: int) -> List[FusedDoc]:
        """CrossEncoder 점수 순으로 상위 top_k건을 반환합니다.

        입력이 비어 있으면 모델을 올리지 않고 바로 빈 리스트를 반환합니다.
        """
        if top_k < 1:
            raise ValueError(f"top_k는 1 이상이어야 합니다 (받은 값: {top_k})")
        if not docs:
            return []

        model = self._load()
        pairs = [(query, d.text) for d in docs]
        scores = model.predict(pairs, batch_size=self.batch_size)

        # 동점일 때 RRF 순위를 유지합니다. 안정 정렬을 쓰면 입력 순서가
        # 보존되고, 입력은 이미 RRF 점수 내림차순입니다.
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
        return [docs[i] for i in order[:top_k]]
