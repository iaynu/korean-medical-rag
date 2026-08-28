"""문서 식별과 BM25 검색 테스트.

원본의 ``text[:80]`` ID가 실제로 충돌한다는 것을 재현하고,
새 구현이 그 충돌을 없앤다는 것을 확인합니다.
"""

from __future__ import annotations

import pytest

from korean_medical_rag.exceptions import DuplicateDocumentIdError, EmptyCorpusError
from korean_medical_rag.retrieval import BM25Retriever, VectorRetriever, make_doc_id


class TestOriginalIdCollision:
    """원본 방식이 깨지는 조건을 명시적으로 남깁니다."""

    @staticmethod
    def _original_get_id(text, meta):
        m = meta or {}
        return m.get("file_id") or m.get("file_name") or text[:80]

    # 실제 진료 지침 문서의 도입부처럼 80자를 넘는 공통 서두.
    INTRO = (
        "본 지침은 대한당뇨병학회가 2023년에 발간한 제2형 당뇨병 진료 권고안으로, "
        "성인 환자의 혈당 관리 목표와 경구 약제 선택 기준, 인슐린 치료 개시 시점을 "
        "근거 수준과 함께 정리한 문서이다. "
    )

    def test_intro_exceeds_id_prefix_length(self):
        """전제 확인: 공통 서두가 80자를 넘어야 충돌이 재현됩니다."""
        assert len(self.INTRO) > 80

    def test_original_collides_on_shared_prefix(self):
        """같은 문서를 청크로 나누면 도입부가 반복되어 ID가 겹칩니다."""
        chunk_a = self.INTRO + "제1장 진단 기준"
        chunk_b = self.INTRO + "제2장 약물 치료"

        assert chunk_a != chunk_b
        assert self._original_get_id(chunk_a, {}) == self._original_get_id(chunk_b, {})

    def test_new_implementation_does_not_collide(self):
        chunk_a = self.INTRO + "제1장 진단 기준"
        chunk_b = self.INTRO + "제2장 약물 치료"

        assert make_doc_id(chunk_a) != make_doc_id(chunk_b)

    def test_original_collides_on_same_file_id(self):
        """같은 파일의 서로 다른 청크가 같은 file_id를 가지면 겹칩니다."""
        meta = {"file_id": "med_001"}
        assert self._original_get_id("청크 1", meta) == self._original_get_id("청크 2", meta)

    def test_new_implementation_separates_chunks(self):
        a = make_doc_id("청크 1", {"file_id": "med_001", "chunk_index": 0})
        b = make_doc_id("청크 2", {"file_id": "med_001", "chunk_index": 1})
        assert a != b
        assert a == "med_001#0"


class TestMakeDocId:
    def test_prefers_metadata_id(self):
        assert make_doc_id("본문", {"file_id": "m1"}) == "m1"

    @pytest.mark.parametrize("key", ["doc_id", "file_id", "chunk_id", "id"])
    def test_accepts_known_id_keys(self, key):
        assert make_doc_id("본문", {key: "X"}) == "X"

    def test_id_key_priority(self):
        """doc_id가 file_id보다 우선합니다."""
        assert make_doc_id("본문", {"file_id": "f", "doc_id": "d"}) == "d"

    def test_ignores_empty_metadata_values(self):
        """빈 문자열이 ID로 쓰이면 모든 문서가 같은 ID를 갖게 됩니다."""
        assert make_doc_id("본문", {"file_id": ""}).startswith("sha1:")
        assert make_doc_id("본문", {"file_id": None}).startswith("sha1:")

    def test_hash_is_stable(self):
        assert make_doc_id("본문") == make_doc_id("본문")

    def test_hash_differs_by_content(self):
        assert make_doc_id("본문 A") != make_doc_id("본문 B")

    def test_no_metadata(self):
        assert make_doc_id("본문", None).startswith("sha1:")


class TestBM25Retriever:
    def test_indexes_corpus(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas)
        assert len(r) == len(medical_corpus)

    def test_empty_corpus_raises(self):
        with pytest.raises(EmptyCorpusError, match="인덱싱할 문서가 없습니다"):
            BM25Retriever([])

    def test_length_mismatch_raises(self, medical_corpus):
        with pytest.raises(ValueError, match="길이가 다릅니다"):
            BM25Retriever(medical_corpus, [{"file_id": "a"}])

    def test_duplicate_ids_raise_by_default(self):
        """조용히 덮어쓰는 대신 인덱싱 시점에 실패합니다."""
        texts = ["청크 1", "청크 2"]
        metas = [{"file_id": "same"}, {"file_id": "same"}]
        with pytest.raises(DuplicateDocumentIdError, match="문서 ID가 1건 충돌"):
            BM25Retriever(texts, metas)

    def test_duplicate_error_suggests_fix(self):
        texts = ["청크 1", "청크 2"]
        metas = [{"file_id": "same"}, {"file_id": "same"}]
        with pytest.raises(DuplicateDocumentIdError, match="chunk_index"):
            BM25Retriever(texts, metas)

    def test_keep_first_mode(self):
        texts = ["청크 1", "청크 2"]
        metas = [{"file_id": "same"}, {"file_id": "same"}]
        r = BM25Retriever(texts, metas, on_duplicate="keep_first")
        assert len(r) == 1
        assert r.duplicate_count == 1

    def test_invalid_on_duplicate(self, medical_corpus):
        with pytest.raises(ValueError, match="on_duplicate는"):
            BM25Retriever(medical_corpus, on_duplicate="ignore")

    def test_retrieves_relevant_doc(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="ngram")
        results = r.retrieve("당뇨병 합병증", k=3)
        assert results
        assert "당뇨병" in results[0].text

    def test_josa_query_still_matches(self, medical_corpus, corpus_metas):
        """조사가 붙은 질의로도 검색되어야 합니다. 원본이 실패하던 지점입니다."""
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="ngram")
        results = r.retrieve("당뇨병의 합병증은?", k=3)
        assert results, "조사 붙은 질의에서 결과 없음"
        assert any("당뇨병" in d.text for d in results)

    def test_whitespace_mode_fails_on_josa(self, medical_corpus, corpus_metas):
        """대조군: 원본 방식은 같은 질의에서 관련 문서를 놓칩니다."""
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="whitespace")
        results = r.retrieve("당뇨병의 합병증은?", k=3)
        # 코퍼스에 "당뇨병의"라는 표면형이 있는 문서 하나만 걸리거나 아예 없습니다.
        assert len(results) <= 1

    def test_zero_score_docs_excluded(self, medical_corpus, corpus_metas):
        """관련 없는 질의는 빈 결과. 0점 문서를 RRF에 넣지 않기 위함입니다."""
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="ngram")
        results = r.retrieve("zzzqqq", k=5)
        assert results == []

    def test_empty_query_returns_empty(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas)
        assert r.retrieve("", k=5) == []
        assert r.retrieve("!!!", k=5) == []

    def test_k_limits_results(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="ngram")
        assert len(r.retrieve("당뇨병 환자 치료", k=2)) <= 2

    def test_invalid_k(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas)
        with pytest.raises(ValueError, match="k는 1 이상"):
            r.retrieve("당뇨병", k=0)

    def test_results_carry_metadata(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="ngram")
        results = r.retrieve("당뇨병", k=1)
        assert results[0].metadata["file_id"].startswith("med_")

    def test_results_are_rank_ordered(self, medical_corpus, corpus_metas):
        r = BM25Retriever(medical_corpus, corpus_metas, tokenizer_mode="ngram")
        results = r.retrieve("당뇨병 합병증", k=5)
        scores = [d.score for d in results]
        assert scores == sorted(scores, reverse=True)

    def test_defaults_metadata_when_omitted(self, medical_corpus):
        r = BM25Retriever(medical_corpus, tokenizer_mode="ngram")
        assert len(r) == len(medical_corpus)


class FakeCollection:
    """Chroma 컬렉션 흉내. chromadb 없이 VectorRetriever를 테스트합니다."""

    def __init__(self, documents, metadatas, distances):
        self._d = (documents, metadatas, distances)
        self.last_call = None

    def query(self, query_embeddings, n_results, include):
        self.last_call = {"n_results": n_results, "include": include}
        docs, metas, dists = self._d
        return {
            "documents": [docs[:n_results]],
            "metadatas": [metas[:n_results]],
            "distances": [dists[:n_results]],
        }


class TestVectorRetriever:
    def _make(self):
        col = FakeCollection(
            documents=["당뇨병 개요", "고혈압 개요"],
            metadatas=[{"file_id": "m1"}, {"file_id": "m2"}],
            distances=[0.1, 0.4],
        )
        return col, VectorRetriever(col, embed_fn=lambda texts: [[0.1, 0.2]])

    def test_returns_docs_in_order(self):
        _, r = self._make()
        results = r.retrieve("당뇨병", k=2)
        assert [d.doc_id for d in results] == ["m1", "m2"]

    def test_distance_flipped_to_similarity(self):
        """거리는 작을수록 좋고 점수는 클수록 좋게 통일합니다."""
        _, r = self._make()
        results = r.retrieve("당뇨병", k=2)
        assert results[0].score == pytest.approx(-0.1)
        assert results[0].score > results[1].score

    def test_passes_k_to_collection(self):
        col, r = self._make()
        r.retrieve("당뇨병", k=1)
        assert col.last_call["n_results"] == 1

    def test_invalid_k(self):
        _, r = self._make()
        with pytest.raises(ValueError, match="k는 1 이상"):
            r.retrieve("당뇨병", k=0)

    def test_handles_null_metadata(self):
        col = FakeCollection(["본문"], [None], [0.2])
        r = VectorRetriever(col, embed_fn=lambda texts: [[0.1]])
        results = r.retrieve("q", k=1)
        assert results[0].doc_id.startswith("sha1:")
