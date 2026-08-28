"""검색기 구현과 문서 식별.

이 모듈은 무거운 의존성을 요구하지 않습니다. BM25는 ``rank_bm25`` 만,
벡터 검색은 호출자가 넘겨주는 객체에만 의존합니다. 덕분에 검색 로직을
GPU나 Chroma 없이 테스트할 수 있습니다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence

from .exceptions import DuplicateDocumentIdError, EmptyCorpusError, RetrievalError
from .fusion import RetrievedDoc
from .tokenization import DEFAULT_NGRAM_SIZE, get_tokenizer

__all__ = [
    "make_doc_id",
    "Retriever",
    "BM25Retriever",
    "VectorRetriever",
]

# 메타데이터에서 문서 식별자로 쓸 키를 우선순위대로 봅니다.
ID_KEYS = ("doc_id", "file_id", "chunk_id", "id")


def make_doc_id(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """문서의 안정적인 식별자를 만듭니다.

    원본 노트북은 이렇게 했습니다::

        return m.get("file_id") or m.get("file_name") or text[:80]

    메타데이터에 ID가 없으면 **본문 앞 80자**를 ID로 썼습니다. 의학
    문서는 같은 원본을 청크로 나누면 도입부가 그대로 반복되는 경우가
    많아서, 서로 다른 청크가 같은 ID를 갖는 일이 실제로 생깁니다.
    그러면 융합 단계의 딕셔너리에서 한쪽이 조용히 덮어써졌습니다.
    검색 후보가 사라진 것을 알 방법이 없었습니다.

    여기서는 두 가지를 바꿉니다.

    1. 메타데이터 ID가 있으면 그대로 씁니다. 단, 청크 인덱스가 있으면
       함께 붙여 같은 파일의 서로 다른 청크를 구분합니다.
    2. 없으면 **본문 전체의 SHA-1** 을 씁니다. 앞 80자가 같아도 본문이
       다르면 다른 ID가 나옵니다.

    >>> make_doc_id("당뇨병 개요", {"file_id": "med_001"})
    'med_001'
    >>> make_doc_id("당뇨병 개요", {"file_id": "med_001", "chunk_index": 3})
    'med_001#3'

    앞부분이 같고 뒤가 다른 두 청크는 다른 ID를 받습니다.

    >>> a = "같은 도입부입니다. " * 10 + "결론 A"
    >>> b = "같은 도입부입니다. " * 10 + "결론 B"
    >>> a[:80] == b[:80]
    True
    >>> make_doc_id(a) == make_doc_id(b)
    False

    같은 본문은 항상 같은 ID입니다. 실행마다 달라지면 결과를 비교할 수
    없기 때문에 해시를 씁니다.

    >>> make_doc_id("당뇨병 개요") == make_doc_id("당뇨병 개요")
    True
    >>> make_doc_id("당뇨병 개요")
    'sha1:373ae1bc36b46bc50dcba6b666107785a3af8067'
    """
    meta = metadata or {}
    for key in ID_KEYS:
        value = meta.get(key)
        if value not in (None, ""):
            base = str(value)
            chunk = meta.get("chunk_index", meta.get("chunk_id"))
            if chunk not in (None, "") and key != "chunk_id":
                return f"{base}#{chunk}"
            return base

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


class Retriever(Protocol):
    """검색기 인터페이스.

    파이프라인은 이 프로토콜만 알면 됩니다. 새 검색기를 붙일 때
    파이프라인 코드를 고칠 필요가 없고, 테스트에서는 가짜 검색기를
    끼워 넣을 수 있습니다.
    """

    name: str

    def retrieve(self, query: str, k: int) -> List[RetrievedDoc]:
        """질의에 대해 상위 k건을 순위대로 반환합니다."""
        ...


@dataclass
class _IndexedCorpus:
    doc_ids: List[str]
    texts: List[str]
    metadatas: List[Mapping[str, Any]]


class BM25Retriever:
    """BM25 키워드 검색기.

    원본과 다른 점은 토크나이저입니다. 자세한 이유는
    :mod:`korean_medical_rag.tokenization` 을 참고하세요.
    """

    name = "bm25"

    def __init__(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        tokenizer_mode: str = "auto",
        ngram_size: int = DEFAULT_NGRAM_SIZE,
        on_duplicate: str = "raise",
    ) -> None:
        """코퍼스로 BM25 인덱스를 만듭니다.

        Args:
            texts: 문서 본문.
            metadatas: 문서별 메타데이터. 없으면 빈 딕셔너리로 채웁니다.
            tokenizer_mode: auto / morph / ngram / whitespace.
            ngram_size: ngram 모드일 때 n.
            on_duplicate: ID가 충돌할 때 ``raise`` 또는 ``keep_first``.

        Raises:
            EmptyCorpusError: 문서가 없을 때.
            DuplicateDocumentIdError: ID 충돌이 있고 on_duplicate='raise'일 때.
        """
        if not texts:
            raise EmptyCorpusError("인덱싱할 문서가 없습니다")

        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError(
                f"texts({len(texts)})와 metadatas({len(metadatas)})의 길이가 다릅니다"
            )
        if on_duplicate not in ("raise", "keep_first"):
            raise ValueError(f"on_duplicate는 raise 또는 keep_first여야 합니다: {on_duplicate!r}")

        self.tokenizer_mode = tokenizer_mode
        self._tokenize = get_tokenizer(tokenizer_mode, ngram_size)

        seen: Dict[str, int] = {}
        kept_ids: List[str] = []
        kept_texts: List[str] = []
        kept_metas: List[Mapping[str, Any]] = []
        duplicates: List[str] = []

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            doc_id = make_doc_id(text, meta)
            if doc_id in seen:
                # 어느 쪽 정책이든 첫 문서를 남깁니다. 차이는 아래에서
                # 예외를 던지느냐 계수만 남기느냐입니다.
                duplicates.append(doc_id)
                continue
            seen[doc_id] = i
            kept_ids.append(doc_id)
            kept_texts.append(text)
            kept_metas.append(meta or {})

        if duplicates and on_duplicate == "raise":
            sample = ", ".join(duplicates[:5])
            more = ""
            if len(duplicates) > 5:
                more = f" 외 {len(duplicates) - 5}건"
            raise DuplicateDocumentIdError(
                f"문서 ID가 {len(duplicates)}건 충돌합니다: {sample}{more}. "
                f"메타데이터에 chunk_index를 넣거나 on_duplicate='keep_first'를 쓰세요."
            )

        self.duplicate_count = len(duplicates)
        self._corpus = _IndexedCorpus(kept_ids, kept_texts, list(kept_metas))

        from rank_bm25 import BM25Okapi

        tokenized = [self._tokenize(t) for t in kept_texts]
        self._bm25 = BM25Okapi(tokenized)

    def __len__(self) -> int:
        return len(self._corpus.doc_ids)

    def retrieve(self, query: str, k: int) -> List[RetrievedDoc]:
        """BM25 점수 상위 k건을 반환합니다.

        점수가 모두 0이면 (질의 토큰이 코퍼스에 하나도 없으면) 빈 리스트를
        반환합니다. 0점 문서를 순위에 넣으면 RRF가 무의미한 문서에
        점수를 주게 됩니다.
        """
        if k < 1:
            raise ValueError(f"k는 1 이상이어야 합니다 (받은 값: {k})")

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

        return [
            RetrievedDoc(
                doc_id=self._corpus.doc_ids[i],
                text=self._corpus.texts[i],
                metadata=self._corpus.metadatas[i],
                score=float(scores[i]),
            )
            for i in ranked
            if scores[i] > 0
        ]


class VectorRetriever:
    """Chroma 컬렉션 기반 벡터 검색기.

    컬렉션과 임베딩 함수를 주입받습니다. 이 클래스 자체는 chromadb나
    sentence-transformers를 import하지 않으므로, 테스트에서 가짜 객체를
    넣어 검색 로직만 검증할 수 있습니다.
    """

    name = "vector"

    def __init__(
        self,
        collection: Any,
        embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    ) -> None:
        self._collection = collection
        self._embed = embed_fn

    def retrieve(self, query: str, k: int) -> List[RetrievedDoc]:
        if k < 1:
            raise ValueError(f"k는 1 이상이어야 합니다 (받은 값: {k})")

        embeddings = self._embed([query])
        if not len(embeddings):
            raise RetrievalError("질의 임베딩이 비어 있습니다")

        result = self._collection.query(
            query_embeddings=[list(embeddings[0])],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0] or [{}] * len(docs)
        dists = result["distances"][0]

        out: List[RetrievedDoc] = []
        for text, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            out.append(
                RetrievedDoc(
                    doc_id=make_doc_id(text, meta),
                    text=text,
                    metadata=meta,
                    # 거리는 작을수록 가깝습니다. 부호를 뒤집어 "클수록 좋음"으로
                    # 통일합니다. RRF는 순위만 보므로 이 값 자체는 로깅용입니다.
                    score=-float(dist),
                )
            )
        return out
