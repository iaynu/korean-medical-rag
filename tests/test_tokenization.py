"""한국어 토크나이저 테스트.

핵심 주장은 "공백 분리로는 한국어 BM25가 동작하지 않는다"입니다.
그 주장을 실제로 검증하는 테스트를 먼저 둡니다.
"""

from __future__ import annotations

import pytest

from korean_medical_rag.tokenization import (
    char_ngrams,
    describe_mode,
    get_tokenizer,
    ngram_tokenize,
    script_runs,
    whitespace_tokenize,
)


class TestJosaProblem:
    """조사 때문에 공백 분리가 실패하는 상황."""

    @pytest.mark.parametrize(
        "query,document",
        [
            ("당뇨병 합병증", "당뇨병의 합병증은 심각하다"),
            ("인슐린 분비", "인슐린이 분비되는 과정"),
            ("고혈압 치료", "고혈압을 치료하려면"),
            ("천식 환자", "천식 환자에게는"),
        ],
    )
    def test_whitespace_finds_no_overlap(self, query, document):
        """공백 분리에서는 의미가 같아도 겹치는 토큰이 거의 없습니다."""
        q = set(whitespace_tokenize(query))
        d = set(whitespace_tokenize(document))
        # 조사가 붙은 쪽은 표면형이 달라 매칭이 안 됩니다.
        assert len(q & d) < len(q), (
            f"공백 분리가 예상과 달리 전부 매칭됨: {q & d}"
        )

    @pytest.mark.parametrize(
        "query,document",
        [
            ("당뇨병 합병증", "당뇨병의 합병증은 심각하다"),
            ("인슐린 분비", "인슐린이 분비되는 과정"),
            ("고혈압 치료", "고혈압을 치료하려면"),
            ("천식 환자", "천식 환자에게는"),
        ],
    )
    def test_ngram_finds_overlap(self, query, document):
        """n-gram은 어간이 겹치므로 매칭됩니다."""
        q = set(ngram_tokenize(query))
        d = set(ngram_tokenize(document))
        assert len(q & d) >= 2, f"n-gram 매칭 실패: {q & d}"

    def test_ngram_strictly_better_than_whitespace(self):
        """같은 질의·문서 쌍에서 n-gram이 더 많이 매칭합니다."""
        query = "당뇨병 합병증 관리"
        document = "당뇨병의 합병증을 관리하는 방법"

        ws = len(set(whitespace_tokenize(query)) & set(whitespace_tokenize(document)))
        ng = len(set(ngram_tokenize(query)) & set(ngram_tokenize(document)))

        assert ws == 0
        assert ng > 0


class TestScriptRuns:
    def test_splits_mixed_scripts(self):
        assert script_runs("당뇨병(diabetes) 진단") == ["당뇨병", "diabetes", "진단"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("PM2.5 수치", ["PM2.5", "수치"]),
            ("COVID-19 백신", ["COVID-19", "백신"]),
            ("HbA1c 검사", ["HbA1c", "검사"]),
        ],
    )
    def test_keeps_medical_terms_intact(self, text, expected):
        """의학 용어의 점·하이픈이 토큰을 쪼개지 않아야 합니다."""
        assert script_runs(text) == expected

    def test_drops_punctuation(self):
        assert script_runs("당뇨병!!! 진단???") == ["당뇨병", "진단"]

    def test_handles_non_string(self):
        assert script_runs(None) == []
        assert script_runs(123) == []

    def test_empty_string(self):
        assert script_runs("") == []


class TestCharNgrams:
    def test_bigrams(self):
        assert char_ngrams("당뇨병") == ["당뇨", "뇨병"]

    def test_short_token_kept_whole(self):
        """한 글자 장기 이름이 사라지면 안 됩니다."""
        assert char_ngrams("간") == ["간"]
        assert char_ngrams("위") == ["위"]

    def test_token_equal_to_n(self):
        assert char_ngrams("당뇨", n=2) == ["당뇨"]

    def test_empty_token(self):
        assert char_ngrams("") == []

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n은 1 이상"):
            char_ngrams("당뇨병", n=0)

    def test_trigrams(self):
        assert char_ngrams("합병증세", n=3) == ["합병증", "병증세"]


class TestLatinHandling:
    def test_latin_not_split_into_ngrams(self):
        """라틴 문자는 n-gram으로 쪼개지 않습니다."""
        assert ngram_tokenize("WHO") == ["who"]

    def test_latin_lowercased(self):
        assert ngram_tokenize("Insulin") == ["insulin"]

    def test_case_insensitive_matching(self):
        assert set(ngram_tokenize("WHO 기준")) & set(ngram_tokenize("who 지침"))


class TestGetTokenizer:
    def test_auto_returns_callable(self):
        tok = get_tokenizer("auto")
        assert callable(tok)
        assert tok("당뇨병")

    def test_explicit_modes(self):
        assert get_tokenizer("ngram")("당뇨병") == ["당뇨", "뇨병"]
        assert get_tokenizer("whitespace")("당뇨 병") == ["당뇨", "병"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="알 수 없는 tokenizer_mode"):
            get_tokenizer("morphological")

    def test_ngram_size_honored(self):
        assert get_tokenizer("ngram", n=3)("합병증세") == ["합병증", "병증세"]

    def test_describe_mode_resolves_auto(self):
        """auto가 무엇으로 해석됐는지 항상 알 수 있어야 합니다."""
        assert describe_mode("auto") in (
            "형태소 분석 (kiwipiepy)",
            "문자 n-gram",
        )
        assert describe_mode("whitespace") == "공백 분리 (원본 방식)"


class TestDeterminism:
    def test_same_input_same_output(self):
        """같은 입력은 항상 같은 토큰. 인덱스 재현성의 전제입니다."""
        text = "당뇨병의 합병증 관리 방법"
        assert ngram_tokenize(text) == ngram_tokenize(text)
