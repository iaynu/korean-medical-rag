"""설정 검증 테스트."""

from __future__ import annotations

import pytest

from korean_medical_rag.config import RagConfig
from korean_medical_rag.exceptions import ConfigError


class TestValidation:
    def test_defaults(self):
        c = RagConfig(db_path="/tmp/db")
        assert c.vector_k == 20
        assert c.bm25_k == 20
        assert c.rerank_k == 5
        assert c.rrf_k == 60

    def test_empty_db_path_rejected(self):
        with pytest.raises(ConfigError, match="db_path는 비어 있을 수 없습니다"):
            RagConfig(db_path="")

    @pytest.mark.parametrize("field", ["vector_k", "bm25_k", "rerank_k", "max_new_tokens"])
    def test_non_positive_rejected(self, field):
        with pytest.raises(ConfigError, match="1 이상"):
            RagConfig(db_path="/tmp/db", **{field: 0})

    @pytest.mark.parametrize("field", ["vector_k", "bm25_k", "rerank_k"])
    def test_negative_rejected(self, field):
        with pytest.raises(ConfigError):
            RagConfig(db_path="/tmp/db", **{field: -1})

    def test_float_k_rejected(self):
        with pytest.raises(ConfigError, match="정수여야"):
            RagConfig(db_path="/tmp/db", vector_k=2.5)

    def test_bool_rejected_as_int(self):
        """bool은 int의 서브클래스라 통과할 수 있어 명시적으로 막습니다."""
        with pytest.raises(ConfigError, match="정수여야"):
            RagConfig(db_path="/tmp/db", vector_k=True)

    def test_rerank_k_larger_than_candidates_rejected(self):
        """후보보다 많이 뽑을 수는 없습니다. 원본은 조용히 넘어갔습니다."""
        with pytest.raises(ConfigError, match="후보 총합"):
            RagConfig(db_path="/tmp/db", vector_k=2, bm25_k=2, rerank_k=10)

    def test_rerank_k_equal_to_candidates_ok(self):
        RagConfig(db_path="/tmp/db", vector_k=2, bm25_k=3, rerank_k=5)

    def test_invalid_tokenizer_mode(self):
        with pytest.raises(ConfigError, match="tokenizer_mode는"):
            RagConfig(db_path="/tmp/db", tokenizer_mode="mecab")

    @pytest.mark.parametrize("mode", ["auto", "morph", "ngram", "whitespace"])
    def test_valid_tokenizer_modes(self, mode):
        assert RagConfig(db_path="/tmp/db", tokenizer_mode=mode).tokenizer_mode == mode

    def test_invalid_rrf_k(self):
        with pytest.raises(ConfigError, match="rrf_k는"):
            RagConfig(db_path="/tmp/db", rrf_k=0)


class TestImmutability:
    def test_frozen(self):
        c = RagConfig(db_path="/tmp/db")
        with pytest.raises(Exception):
            c.vector_k = 50


class TestFromMapping:
    def test_builds_from_dict(self):
        c = RagConfig.from_mapping({"db_path": "/tmp/db", "rerank_k": 3})
        assert c.rerank_k == 3

    def test_unknown_key_rejected(self):
        """오타 난 키를 무시하면 기본값으로 실험이 돌아갑니다."""
        with pytest.raises(ConfigError, match="알 수 없는 설정 키: rerank_kk"):
            RagConfig.from_mapping({"db_path": "/tmp/db", "rerank_kk": 3})

    def test_multiple_unknown_keys_listed(self):
        with pytest.raises(ConfigError, match="alpha, beta"):
            RagConfig.from_mapping({"db_path": "/tmp/db", "beta": 1, "alpha": 2})

    def test_validation_still_applies(self):
        with pytest.raises(ConfigError, match="1 이상"):
            RagConfig.from_mapping({"db_path": "/tmp/db", "vector_k": 0})


class TestFusionTopN:
    def test_default_is_none(self):
        assert RagConfig(db_path="/tmp/db").fusion_top_n is None

    def test_accepts_int(self):
        assert RagConfig(db_path="/tmp/db", fusion_top_n=10).fusion_top_n == 10

    def test_smaller_than_rerank_k_rejected(self):
        """재랭커에 3건 넘기고 5건 뽑을 수는 없습니다."""
        with pytest.raises(ConfigError, match="보다 작습니다"):
            RagConfig(db_path="/tmp/db", rerank_k=5, fusion_top_n=3)

    def test_equal_to_rerank_k_ok(self):
        RagConfig(db_path="/tmp/db", rerank_k=5, fusion_top_n=5)

    def test_non_int_rejected(self):
        with pytest.raises(ConfigError, match="fusion_top_n은"):
            RagConfig(db_path="/tmp/db", fusion_top_n=2.5)
