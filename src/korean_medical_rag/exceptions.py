"""이 패키지가 던지는 예외 계층.

호출자가 실패 원인별로 다르게 대응할 수 있도록 나눴습니다.
설정 오류는 즉시 고칠 수 있지만, 검색 결과 없음은 질의를 바꿔야 하고,
백엔드 오류는 재시도 대상입니다.
"""

from __future__ import annotations


class RagError(Exception):
    """이 패키지에서 발생하는 모든 예외의 최상위."""


class ConfigError(RagError):
    """설정값이 유효하지 않을 때."""


class CorpusError(RagError):
    """문서 코퍼스를 읽거나 인덱싱할 수 없을 때."""


class EmptyCorpusError(CorpusError):
    """인덱싱할 문서가 하나도 없을 때."""


class DuplicateDocumentIdError(CorpusError):
    """서로 다른 문서가 같은 ID를 가질 때.

    원본 노트북은 ID 충돌 시 한쪽을 조용히 덮어썼습니다. 검색 후보가
    말없이 사라지는 것보다 인덱싱 시점에 실패하는 편이 낫습니다.
    """


class RetrievalError(RagError):
    """검색 백엔드가 결과를 반환하지 못했을 때."""


class GenerationError(RagError):
    """LLM 생성 단계 실패."""
