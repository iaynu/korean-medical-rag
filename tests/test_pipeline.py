"""파이프라인 오케스트레이션 테스트.

가짜 검색기·재랭커·생성기로 전체 흐름을 GPU 없이 검증합니다.
"""

from __future__ import annotations

import pytest

from korean_medical_rag.config import RagConfig
from korean_medical_rag.pipeline import RagPipeline, RagResult

from conftest import FakeReranker, FakeRetriever, docs


@pytest.fixture
def config():
    return RagConfig(db_path="/tmp/db", vector_k=3, bm25_k=3, rerank_k=2)


@pytest.fixture
def pipeline(config, fake_generator):
    return RagPipeline(
        config=config,
        retrievers=[
            FakeRetriever("vector", docs("A", "C", "D")),
            FakeRetriever("bm25", docs("B", "C", "D")),
        ],
        reranker=None,
        generator=fake_generator,
    )


class TestConstruction:
    def test_requires_retriever(self, config):
        with pytest.raises(ValueError, match="검색기가 최소 하나는"):
            RagPipeline(config=config, retrievers=[])

    def test_rejects_duplicate_retriever_names(self, config):
        """이름이 겹치면 RRF에서 한쪽 결과가 사라집니다."""
        with pytest.raises(ValueError, match="이름이 중복"):
            RagPipeline(
                config=config,
                retrievers=[FakeRetriever("bm25", docs("A")), FakeRetriever("bm25", docs("B"))],
            )


class TestRetrievalWiring:
    def test_passes_configured_k_per_retriever(self, config, fake_generator):
        vec = FakeRetriever("vector", docs("A", "B", "C", "D"))
        bm = FakeRetriever("bm25", docs("A", "B", "C", "D"))
        cfg = RagConfig(db_path="/tmp/db", vector_k=3, bm25_k=2, rerank_k=2)

        RagPipeline(cfg, [vec, bm], generator=fake_generator).answer("질문", 3)

        assert vec.calls == [("질문", 3)]
        assert bm.calls == [("질문", 2)]

    def test_fuses_both_retrievers(self, pipeline):
        result = pipeline.answer("질문", 3)
        # C와 D는 양쪽에, A는 vector에만, B는 bm25에만 있습니다.
        assert {f.doc_id for f in result.fused} == {"A", "B", "C", "D"}

    def test_consensus_doc_ranked_first(self, pipeline):
        result = pipeline.answer("질문", 3)
        assert result.fused[0].doc_id == "C"


class TestReranking:
    def test_without_reranker_uses_rrf_top_k(self, pipeline, config):
        result = pipeline.answer("질문", 3)
        assert len(result.reranked) == config.rerank_k
        assert result.reranked == result.fused[: config.rerank_k]

    def test_reranker_is_called_and_changes_order(self, config, fake_generator):
        reranker = FakeReranker()
        p = RagPipeline(
            config,
            [FakeRetriever("vector", docs("A", "C", "D")), FakeRetriever("bm25", docs("B", "C", "D"))],
            reranker=reranker,
            generator=fake_generator,
        )
        result = p.answer("질문", 3)

        assert reranker.calls == [("질문", config.rerank_k)]
        # FakeReranker는 순서를 뒤집습니다.
        assert result.reranked[0].doc_id != result.fused[0].doc_id

    def test_reranker_skipped_when_no_candidates(self, config, fake_generator):
        """후보가 없으면 재랭커를 호출하지 않습니다. 모델 로딩 낭비를 막습니다."""
        reranker = FakeReranker()
        p = RagPipeline(
            config,
            [FakeRetriever("vector", []), FakeRetriever("bm25", [])],
            reranker=reranker,
            generator=fake_generator,
        )
        result = p.answer("질문", 3)

        assert reranker.calls == []
        assert result.reranked == []


class TestGeneration:
    def test_generator_receives_prompt_with_context(self, pipeline, fake_generator):
        pipeline.answer("당뇨병이란?", 3)
        prompt = fake_generator.prompts[0]
        assert "당뇨병이란?" in prompt
        assert "[참고자료]" in prompt

    def test_answer_returned(self, pipeline):
        assert pipeline.answer("질문", 3).answer == "테스트 답변"

    def test_without_generator_returns_empty_answer(self, config):
        """생성기 없이 검색 단계만 확인할 수 있어야 합니다."""
        p = RagPipeline(config, [FakeRetriever("bm25", docs("A", "B"))])
        result = p.answer("질문", 3)
        assert result.answer == ""
        assert result.prompt

    def test_q_type_reaches_prompt(self, pipeline, fake_generator):
        pipeline.answer("문제", 1)
        assert "객관식" in fake_generator.prompts[0]


class TestResultDiagnostics:
    def test_keeps_intermediate_stages(self, config, fake_generator):
        """어느 단계에서 정답 문서가 빠졌는지 추적할 수 있어야 합니다."""
        p = RagPipeline(
            config,
            [FakeRetriever("vector", docs("A", "C", "D")), FakeRetriever("bm25", docs("B", "C", "D"))],
            reranker=FakeReranker(),
            generator=fake_generator,
        )
        result = p.answer("질문", 3)

        assert len(result.fused) == 4
        assert len(result.reranked) == 2
        assert len(result.fused) > len(result.reranked)

    def test_contributions_show_which_retriever_found_doc(self, pipeline):
        result = pipeline.answer("질문", 3)
        by_id = {f.doc_id: dict(f.contributions) for f in result.fused}
        assert by_id["A"] == {"vector": 1}
        assert by_id["B"] == {"bm25": 1}
        assert set(by_id["C"]) == {"vector", "bm25"}

    def test_retrieved_ids_accessor(self, pipeline):
        result = pipeline.answer("질문", 3)
        assert result.retrieved_ids == [d.doc_id for d in result.reranked]

    def test_timings_recorded(self, pipeline):
        t = pipeline.answer("질문", 3).timings_ms
        assert set(t) == {"retrieve", "fuse", "rerank", "generate"}
        assert all(v >= 0 for v in t.values())

    def test_trace_is_readable(self, pipeline):
        trace = pipeline.answer("질문", 3).trace()
        assert "질문:" in trace
        assert "융합:" in trace
        assert "재랭킹:" in trace


class TestEmptyRetrieval:
    def test_all_empty_does_not_crash(self, config, fake_generator):
        """검색 결과가 없어도 배치가 멈추면 안 됩니다."""
        p = RagPipeline(
            config,
            [FakeRetriever("vector", []), FakeRetriever("bm25", [])],
            generator=fake_generator,
        )
        result = p.answer("질문", 3)
        assert result.fused == []
        assert "(참고자료 없음)" in result.prompt

    def test_one_retriever_empty(self, config, fake_generator):
        p = RagPipeline(
            config,
            [FakeRetriever("vector", docs("A", "B")), FakeRetriever("bm25", [])],
            generator=fake_generator,
        )
        result = p.answer("질문", 3)
        assert [f.doc_id for f in result.fused] == ["A", "B"]


class TestRagResult:
    def test_is_frozen(self):
        r = RagResult("q", 1, "a", [], [], "p")
        with pytest.raises(Exception):
            r.answer = "다른 답"


class TestFusionTopN:
    """재랭커에 넘길 후보를 좁혀 실제로 비용을 줄이는지 확인합니다."""

    def test_none_passes_all_candidates(self, fake_generator):
        cfg = RagConfig(db_path="/tmp/db", vector_k=4, bm25_k=4, rerank_k=2,
                        fusion_top_n=None)
        p = RagPipeline(
            cfg,
            [FakeRetriever("vector", docs("A", "B", "C", "D")),
             FakeRetriever("bm25", docs("E", "F", "G", "H"))],
            generator=fake_generator,
        )
        assert len(p.answer("질문", 3).fused) == 8

    def test_top_n_narrows_candidates(self, fake_generator):
        cfg = RagConfig(db_path="/tmp/db", vector_k=4, bm25_k=4, rerank_k=2,
                        fusion_top_n=3)
        p = RagPipeline(
            cfg,
            [FakeRetriever("vector", docs("A", "B", "C", "D")),
             FakeRetriever("bm25", docs("E", "F", "G", "H"))],
            generator=fake_generator,
        )
        assert len(p.answer("질문", 3).fused) == 3

    def test_reranker_receives_narrowed_set(self, fake_generator):
        """비용 절감의 실체: 재랭커가 보는 후보 수가 실제로 줄어야 합니다."""
        seen = []

        class CountingReranker:
            def rerank(self, query, docs_in, top_k):
                seen.append(len(docs_in))
                return list(docs_in)[:top_k]

        for top_n, expected in [(None, 8), (3, 3)]:
            cfg = RagConfig(db_path="/tmp/db", vector_k=4, bm25_k=4,
                            rerank_k=2, fusion_top_n=top_n)
            RagPipeline(
                cfg,
                [FakeRetriever("vector", docs("A", "B", "C", "D")),
                 FakeRetriever("bm25", docs("E", "F", "G", "H"))],
                reranker=CountingReranker(),
                generator=fake_generator,
            ).answer("질문", 3)

        assert seen == [8, 3]


class TestCollisionReporting:
    def test_collision_surfaced_in_result(self, config, fake_generator):
        from korean_medical_rag.fusion import RetrievedDoc

        p = RagPipeline(
            config,
            [FakeRetriever("vector", [RetrievedDoc("dup", "본문 A")]),
             FakeRetriever("bm25", [RetrievedDoc("dup", "본문 B")])],
            generator=fake_generator,
        )
        result = p.answer("질문", 3)

        assert result.id_collisions == {"dup": 2}
        assert "ID 충돌" in result.trace()

    def test_no_collision_no_warning(self, pipeline):
        result = pipeline.answer("질문", 3)
        assert result.id_collisions == {}
        assert "경고" not in result.trace()
