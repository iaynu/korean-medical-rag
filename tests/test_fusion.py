"""RRF 융합 테스트.

원본 merge_candidates가 순위를 버렸다는 주장과, RRF가 그것을 복원한다는
주장을 검증합니다.
"""

from __future__ import annotations

import pytest

from korean_medical_rag.fusion import (
    FusedDoc,
    RetrievedDoc,
    fusion_summary,
    reciprocal_rank_fusion,
)

from conftest import docs


class TestRetrievedDoc:
    def test_requires_doc_id(self):
        with pytest.raises(ValueError, match="doc_id는 비어 있을 수 없습니다"):
            RetrievedDoc(doc_id="", text="본문")

    def test_is_frozen(self):
        d = RetrievedDoc("a", "본문")
        with pytest.raises(Exception):
            d.doc_id = "b"

    def test_default_metadata_is_independent(self):
        """기본 메타데이터가 인스턴스 간에 공유되면 안 됩니다."""
        a = RetrievedDoc("a", "x")
        b = RetrievedDoc("b", "y")
        assert a.metadata is not b.metadata


class TestConsensusWins:
    """RRF의 핵심 성질."""

    def test_agreed_doc_beats_single_top_hit(self):
        """양쪽이 2위로 꼽은 문서가 한쪽만의 1위를 이깁니다."""
        vec = docs("A", "C", "D")
        bm = docs("B", "C", "D")

        fused = reciprocal_rank_fusion({"vector": vec, "bm25": bm}, k=1)

        assert fused[0].doc_id == "C"
        assert dict(fused[0].contributions) == {"vector": 2, "bm25": 2}

    def test_single_list_docs_not_dropped(self):
        """한쪽에만 있는 문서도 후보로 남습니다."""
        vec = docs("A", "C")
        bm = docs("B", "C")

        ids = {f.doc_id for f in reciprocal_rank_fusion({"vector": vec, "bm25": bm})}
        assert ids == {"A", "B", "C"}


class TestScaleInvariance:
    """원본이 점수를 더하지 못했던 이유를 뒤집어 검증합니다."""

    def test_bm25_score_magnitude_does_not_matter(self):
        vec = [RetrievedDoc("A", "a", score=-0.1), RetrievedDoc("B", "b", score=-0.9)]
        small = [RetrievedDoc("B", "b", score=0.001), RetrievedDoc("C", "c", score=0.0005)]
        huge = [RetrievedDoc("B", "b", score=50000.0), RetrievedDoc("C", "c", score=49999.0)]

        a = [f.doc_id for f in reciprocal_rank_fusion({"v": vec, "b": small})]
        b = [f.doc_id for f in reciprocal_rank_fusion({"v": vec, "b": huge})]

        assert a == b

    def test_naive_score_sum_would_differ(self):
        """대조: 점수를 그냥 더하면 스케일 큰 쪽이 지배합니다."""
        vec = {"A": -0.1, "B": -0.9}
        bm = {"B": 50000.0, "C": 49999.0}

        summed = {}
        for d, s in list(vec.items()) + list(bm.items()):
            summed[d] = summed.get(d, 0.0) + s

        order = [d for d, _ in sorted(summed.items(), key=lambda kv: -kv[1])]
        # BM25 점수만으로 순위가 결정되고 벡터 신호는 묻힙니다.
        assert order[0] == "B"
        assert order[1] == "C"
        assert order[-1] == "A"


class TestDuplicateHandling:
    def test_same_doc_twice_in_one_list_counted_once(self):
        """중복 반환이 점수를 부풀리면 안 됩니다."""
        dup = [RetrievedDoc("A", "a"), RetrievedDoc("A", "a"), RetrievedDoc("B", "b")]
        fused = reciprocal_rank_fusion({"v": dup}, k=1)

        assert [f.doc_id for f in fused] == ["A", "B"]
        assert dict(fused[0].contributions) == {"v": 1}
        assert fused[0].rrf_score == pytest.approx(1 / 2)

    def test_duplicate_does_not_inflate_above_consensus(self):
        """중복 3회가 양쪽 합의를 이기면 안 됩니다."""
        spam = [RetrievedDoc("SPAM", "s")] * 3 + [RetrievedDoc("X", "x")]
        other = [RetrievedDoc("X", "x")]

        fused = reciprocal_rank_fusion({"v": spam, "b": other}, k=1)
        assert fused[0].doc_id == "X"


class TestDeterminism:
    def test_ties_broken_by_doc_id(self):
        """동점 순서가 실행마다 달라지면 결과를 비교할 수 없습니다."""
        a = reciprocal_rank_fusion({"v": docs("B", "A")}, k=1)
        b = reciprocal_rank_fusion({"v": docs("B", "A")}, k=1)
        assert [f.doc_id for f in a] == [f.doc_id for f in b]

    def test_tied_scores_sorted_by_id(self):
        vec = docs("Z")
        bm = docs("A")
        fused = reciprocal_rank_fusion({"v": vec, "b": bm}, k=1)
        assert fused[0].rrf_score == fused[1].rrf_score
        assert [f.doc_id for f in fused] == ["A", "Z"]


class TestEdgeCases:
    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []

    def test_all_lists_empty(self):
        assert reciprocal_rank_fusion({"v": [], "b": []}) == []

    def test_one_list_empty(self):
        """한 검색기가 결과를 못 내도 다른 쪽 결과는 살아야 합니다."""
        fused = reciprocal_rank_fusion({"v": docs("A", "B"), "b": []})
        assert [f.doc_id for f in fused] == ["A", "B"]

    def test_top_n_truncates(self):
        fused = reciprocal_rank_fusion({"v": docs("A", "B", "C")}, top_n=2)
        assert len(fused) == 2

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k는 1 이상"):
            reciprocal_rank_fusion({"v": docs("A")}, k=0)

    def test_invalid_top_n(self):
        with pytest.raises(ValueError, match="top_n은 1 이상"):
            reciprocal_rank_fusion({"v": docs("A")}, top_n=0)

    def test_three_retrievers(self):
        """검색기를 늘려도 코드 변경 없이 동작합니다."""
        fused = reciprocal_rank_fusion(
            {"v": docs("A", "X"), "b": docs("B", "X"), "c": docs("C", "X")}, k=1
        )
        assert fused[0].doc_id == "X"
        assert len(fused[0].contributions) == 3


class TestRrfK:
    def test_larger_k_flattens_ranking(self):
        """k가 커지면 상위·하위 점수 차가 줄어듭니다."""
        d = docs("A", "B", "C")
        tight = reciprocal_rank_fusion({"v": d}, k=1)
        flat = reciprocal_rank_fusion({"v": d}, k=1000)

        tight_gap = tight[0].rrf_score - tight[-1].rrf_score
        flat_gap = flat[0].rrf_score - flat[-1].rrf_score
        assert flat_gap < tight_gap


class TestFusionSummary:
    def test_counts_overlap_and_singles(self):
        fused = reciprocal_rank_fusion({"vector": docs("A", "B"), "bm25": docs("B", "C")})
        assert fusion_summary(fused) == "후보 3건 (양쪽 1건, vector 단독 1건, bm25 단독 1건)"

    def test_empty(self):
        assert fusion_summary([]) == "후보 0건"


class TestFusedDocAccessors:
    def test_proxies_doc_fields(self):
        f = FusedDoc(RetrievedDoc("a", "본문 a"), 0.5, {"v": 1})
        assert f.doc_id == "a"
        assert f.text == "본문 a"


class TestIdCollisionDetection:
    """벡터 검색기는 인덱싱 시점이 없어 융합이 마지막 방어선입니다."""

    def test_detects_same_id_different_text(self):
        from korean_medical_rag.fusion import detect_id_collisions

        a = RetrievedDoc("dup", "본문 하나")
        b = RetrievedDoc("dup", "본문 둘")
        assert detect_id_collisions({"v": [a], "b": [b]}) == {"dup": 2}

    def test_same_id_same_text_is_not_collision(self):
        """두 검색기가 같은 문서를 찾은 정상적인 경우."""
        from korean_medical_rag.fusion import detect_id_collisions

        a = RetrievedDoc("same", "동일 본문")
        b = RetrievedDoc("same", "동일 본문")
        assert detect_id_collisions({"v": [a], "b": [b]}) == {}

    def test_no_collisions_returns_empty(self):
        from korean_medical_rag.fusion import detect_id_collisions

        assert detect_id_collisions({"v": docs("A", "B")}) == {}

    def test_collision_within_single_retriever(self):
        """한 검색기 안에서도 충돌할 수 있습니다."""
        from korean_medical_rag.fusion import detect_id_collisions

        ranked = [RetrievedDoc("x", "본문 1"), RetrievedDoc("x", "본문 2")]
        assert detect_id_collisions({"v": ranked}) == {"x": 2}

    def test_fusion_still_collapses_but_now_reported(self):
        """융합은 여전히 합칩니다. 달라진 건 조용하지 않다는 점입니다."""
        from korean_medical_rag.fusion import detect_id_collisions

        lists = {"v": [RetrievedDoc("dup", "A")], "b": [RetrievedDoc("dup", "B")]}
        fused = reciprocal_rank_fusion(lists)

        assert len(fused) == 1
        assert detect_id_collisions(lists) == {"dup": 2}
