"""질문 유형별 프롬프트 구성.

평가셋의 질문은 세 종류입니다. 유형마다 기대하는 답변 형식이 달라서
프롬프트를 나눕니다. 객관식에 서술형 프롬프트를 쓰면 모델이 설명을
늘어놓고, 채점 로직이 선택지 번호를 못 찾습니다.

원본 노트북은 이 로직을 ``build_prompt`` 하나에 if/elif로 담고 tokenizer를
인자로 받았습니다. 여기서는 프롬프트 문자열 구성과 chat template 적용을
분리했습니다. 프롬프트 내용은 tokenizer 없이 테스트할 수 있어야 하고,
실제로 프롬프트를 고칠 때마다 GPU를 띄우고 싶지는 않기 때문입니다.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Sequence

from .fusion import FusedDoc

__all__ = [
    "QuestionType",
    "SYSTEM_PROMPT",
    "build_user_content",
    "build_context",
    "build_messages",
    "build_prompt",
]


class QuestionType(IntEnum):
    """평가셋의 q_type 값.

    >>> QuestionType(1).name
    'MULTIPLE_CHOICE'
    >>> QuestionType.SHORT_ANSWER.value
    2
    """

    MULTIPLE_CHOICE = 1
    SHORT_ANSWER = 2
    DESCRIPTIVE = 3


SYSTEM_PROMPT = """너는 의료 정보 안내용 챗봇이다.
확정적인 진단을 내리지 말고, 일반적인 의학 정보를 바탕으로 설명한다.
근거가 불충분하면 추측하지 말고 모른다고 말한다.
위험 신호가 보이면 반드시 의료기관 방문을 권한다."""


_MULTIPLE_CHOICE = """다음은 객관식 의료 문제이다.

반드시 정답에 해당하는 선택지를 "번호) 선택지 내용" 형식으로만 답하라.
질문과 선택지를 다 읽고 가장 적절한 답을 고르라.
다른 설명은 절대 하지 말라.

질문:
{question}"""

_SHORT_ANSWER = """다음 질문에 대해 단답으로 답하라.

규칙:
단어 또는 짧은 구절로만 답하라.
불필요한 설명을 하지 말라.
근거가 불충분하면 "모르겠습니다"라고만 답하라.

질문:
{question}"""

_DESCRIPTIVE = """다음 질문에 대해 일반적인 의학 정보를 바탕으로 설명하라.

규칙:
핵심 내용에 대해 한 문장으로 간결히 답하라.
불필요한 설명을 하지 말라.
근거가 불충분하면 "모르겠습니다"라고만 답하라.

질문:
{question}"""

_TEMPLATES: Dict[QuestionType, str] = {
    QuestionType.MULTIPLE_CHOICE: _MULTIPLE_CHOICE,
    QuestionType.SHORT_ANSWER: _SHORT_ANSWER,
    QuestionType.DESCRIPTIVE: _DESCRIPTIVE,
}


def build_user_content(question: str, q_type: int | QuestionType) -> str:
    """질문 유형에 맞는 사용자 메시지를 만듭니다.

    Raises:
        ValueError: 질문이 비어 있거나 q_type이 1/2/3이 아닐 때.

    원본은 ``else`` 로 떨어지는 모든 값을 서술형으로 처리했습니다.
    q_type에 4가 들어와도 조용히 서술형으로 돌았고, 채점이 어긋난
    이유를 찾기 어려웠습니다. 여기서는 거부합니다.

    >>> print(build_user_content("당뇨병의 정의는?", 2))
    다음 질문에 대해 단답으로 답하라.
    <BLANKLINE>
    규칙:
    단어 또는 짧은 구절로만 답하라.
    불필요한 설명을 하지 말라.
    근거가 불충분하면 "모르겠습니다"라고만 답하라.
    <BLANKLINE>
    질문:
    당뇨병의 정의는?

    >>> build_user_content("질문", 4)
    Traceback (most recent call last):
        ...
    ValueError: q_type은 1(객관식)/2(단답형)/3(서술형) 중 하나여야 합니다 (받은 값: 4)
    >>> build_user_content("   ", 1)
    Traceback (most recent call last):
        ...
    ValueError: question이 비어 있습니다
    """
    if not question or not question.strip():
        raise ValueError("question이 비어 있습니다")
    try:
        qt = QuestionType(int(q_type))
    except (ValueError, TypeError):
        raise ValueError(
            f"q_type은 1(객관식)/2(단답형)/3(서술형) 중 하나여야 합니다 (받은 값: {q_type})"
        ) from None
    return _TEMPLATES[qt].format(question=question.strip())


def build_context(docs: Sequence[FusedDoc], max_chars: int | None = None) -> str:
    """검색된 문서를 번호 매긴 참고자료 블록으로 만듭니다.

    번호를 붙이는 이유는 모델이 "[2]에 따르면" 처럼 인용할 수 있게 하고,
    답변을 사람이 검토할 때 어느 문서에서 나온 말인지 추적하기 위해서입니다.

    Args:
        docs: 융합·재랭킹을 거친 문서들.
        max_chars: 문서 하나당 본문 길이 상한. None이면 자르지 않습니다.

    >>> from .fusion import RetrievedDoc, FusedDoc
    >>> d1 = FusedDoc(RetrievedDoc("m1", "인슐린은 혈당을 낮춘다", {"file_id": "m1"}), 0.5, {"vector": 1})
    >>> d2 = FusedDoc(RetrievedDoc("m2", "제2형 당뇨병 개요", {"file_id": "m2"}), 0.4, {"bm25": 1})
    >>> print(build_context([d1, d2]))
    [1] (m1) 인슐린은 혈당을 낮춘다
    <BLANKLINE>
    [2] (m2) 제2형 당뇨병 개요

    >>> print(build_context([d1], max_chars=6))
    [1] (m1) 인슐린은 혈…

    문서가 없으면 그렇다고 명시합니다. 빈 문자열을 넣으면 모델이
    참고자료가 있다고 착각하고 지어낼 수 있습니다.

    >>> build_context([])
    '(참고자료 없음)'
    """
    if not docs:
        return "(참고자료 없음)"

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.doc.metadata or {}
        label = meta.get("file_id") or meta.get("file_name") or doc.doc_id
        text = doc.text
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "…"
        parts.append(f"[{i}] ({label}) {text}")
    return "\n\n".join(parts)


def build_messages(
    question: str,
    q_type: int | QuestionType,
    docs: Sequence[FusedDoc],
    max_doc_chars: int | None = None,
) -> List[Dict[str, str]]:
    """chat template에 넣을 messages 리스트를 만듭니다.

    참고자료를 시스템이 아니라 사용자 메시지에 붙입니다. 시스템 메시지는
    역할 지시에만 쓰고, 질의마다 달라지는 내용은 사용자 턴에 두는 편이
    모델이 지시와 자료를 헷갈릴 여지가 적습니다.

    >>> msgs = build_messages("당뇨병이란?", 3, [])
    >>> [m["role"] for m in msgs]
    ['system', 'user']
    >>> "(참고자료 없음)" in msgs[1]["content"]
    True
    """
    user_content = build_user_content(question, q_type)
    context = build_context(docs, max_doc_chars)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_content}\n\n[참고자료]\n{context}"},
    ]


def build_prompt(
    tokenizer: Any,
    question: str,
    q_type: int | QuestionType,
    docs: Sequence[FusedDoc] = (),
    max_doc_chars: int | None = None,
) -> str:
    """최종 프롬프트 문자열. tokenizer가 chat template을 지원하면 사용합니다.

    tokenizer가 None이면 template 없이 평문으로 조립합니다. 프롬프트
    내용을 확인하는 데 모델 로딩이 필요 없게 하기 위한 경로입니다.

    >>> p = build_prompt(None, "당뇨병이란?", 3)
    >>> p.startswith("너는 의료 정보 안내용 챗봇이다.")
    True
    >>> p.endswith("답변:")
    True
    """
    messages = build_messages(question, q_type, docs, max_doc_chars)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    body = "\n\n".join(m["content"] for m in messages)
    return f"{body}\n\n답변:"
