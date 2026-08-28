"""프롬프트 구성 테스트."""

from __future__ import annotations

import pytest

from korean_medical_rag.fusion import FusedDoc, RetrievedDoc
from korean_medical_rag.prompts import (
    QuestionType,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    build_prompt,
    build_user_content,
)


def fused(doc_id, text, **meta):
    return FusedDoc(RetrievedDoc(doc_id, text, meta or {"file_id": doc_id}), 0.5, {"v": 1})


class TestQuestionType:
    def test_values_match_dataset(self):
        assert QuestionType.MULTIPLE_CHOICE == 1
        assert QuestionType.SHORT_ANSWER == 2
        assert QuestionType.DESCRIPTIVE == 3


class TestBuildUserContent:
    @pytest.mark.parametrize(
        "q_type,marker",
        [
            (1, "객관식"),
            (2, "단답"),
            (3, "설명하라"),
        ],
    )
    def test_type_specific_instructions(self, q_type, marker):
        assert marker in build_user_content("질문입니다", q_type)

    def test_multiple_choice_forbids_explanation(self):
        """객관식에서 설명이 나오면 채점이 깨집니다."""
        content = build_user_content("문제", 1)
        assert "다른 설명은 절대 하지 말라" in content

    def test_question_embedded(self):
        assert "당뇨병의 정의는?" in build_user_content("당뇨병의 정의는?", 2)

    def test_question_stripped(self):
        assert build_user_content("  질문  ", 1).endswith("질문")

    def test_accepts_enum(self):
        assert build_user_content("질문", QuestionType.SHORT_ANSWER) == build_user_content(
            "질문", 2
        )

    @pytest.mark.parametrize("bad", [0, 4, 99, -1])
    def test_out_of_range_rejected(self, bad):
        """원본은 else로 떨어뜨려 조용히 서술형 처리했습니다."""
        with pytest.raises(ValueError, match="q_type은"):
            build_user_content("질문", bad)

    def test_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="q_type은"):
            build_user_content("질문", "객관식")

    @pytest.mark.parametrize("bad", ["", "   ", "\n"])
    def test_empty_question_rejected(self, bad):
        with pytest.raises(ValueError, match="question이 비어 있습니다"):
            build_user_content(bad, 1)


class TestBuildContext:
    def test_numbers_documents(self):
        ctx = build_context([fused("m1", "본문 하나"), fused("m2", "본문 둘")])
        assert "[1]" in ctx and "[2]" in ctx

    def test_includes_source_label(self):
        """답변 검토 시 출처 추적을 위해 라벨을 붙입니다."""
        assert "(m1)" in build_context([fused("m1", "본문")])

    def test_falls_back_to_doc_id(self):
        doc = FusedDoc(RetrievedDoc("abc", "본문", {}), 0.1, {"v": 1})
        assert "(abc)" in build_context([doc])

    def test_prefers_file_name_when_no_file_id(self):
        doc = FusedDoc(RetrievedDoc("x", "본문", {"file_name": "가이드.pdf"}), 0.1, {"v": 1})
        assert "(가이드.pdf)" in build_context([doc])

    def test_empty_docs_states_absence(self):
        """빈 문자열을 주면 모델이 자료가 있다고 착각할 수 있습니다."""
        assert build_context([]) == "(참고자료 없음)"

    def test_truncates_long_documents(self):
        ctx = build_context([fused("m1", "가" * 100)], max_chars=10)
        assert "…" in ctx
        assert len(ctx) < 60

    def test_no_truncation_by_default(self):
        long_text = "가" * 200
        assert long_text in build_context([fused("m1", long_text)])


class TestBuildMessages:
    def test_system_and_user_roles(self):
        msgs = build_messages("질문", 3, [])
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_system_prompt_used(self):
        assert build_messages("질문", 3, [])[0]["content"] == SYSTEM_PROMPT

    def test_context_in_user_turn_not_system(self):
        """질의마다 달라지는 자료는 사용자 턴에 둡니다."""
        msgs = build_messages("질문", 3, [fused("m1", "특이한본문")])
        assert "특이한본문" in msgs[1]["content"]
        assert "특이한본문" not in msgs[0]["content"]

    def test_context_header_present(self):
        assert "[참고자료]" in build_messages("질문", 3, [])[1]["content"]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "<TEMPLATE>" + "|".join(m["role"] for m in messages)


class TestBuildPrompt:
    def test_uses_chat_template_when_available(self):
        p = build_prompt(FakeTokenizer(), "질문", 3)
        assert p == "<TEMPLATE>system|user"

    def test_plain_text_without_tokenizer(self):
        """모델 없이 프롬프트를 확인할 수 있어야 합니다."""
        p = build_prompt(None, "질문", 3)
        assert p.startswith(SYSTEM_PROMPT)
        assert p.endswith("답변:")

    def test_tokenizer_without_template_falls_back(self):
        p = build_prompt(object(), "질문", 3)
        assert p.endswith("답변:")

    def test_documents_included(self):
        p = build_prompt(None, "질문", 3, [fused("m1", "고유한본문내용")])
        assert "고유한본문내용" in p

    def test_validation_propagates(self):
        with pytest.raises(ValueError, match="q_type은"):
            build_prompt(None, "질문", 9)
