"""CLI 테스트."""

from __future__ import annotations

import json

import pytest

from korean_medical_rag.cli import build_parser, main


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    records = [
        {"text": "당뇨병의 합병증은 신증과 망막병증이다.", "file_id": "m0"},
        {"text": "고혈압 환자는 저염식이 필요하다.", "file_id": "m1"},
        {"text": "인슐린은 췌장에서 분비된다.", "file_id": "m2"},
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


class TestParser:
    def test_question_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_defaults(self):
        args = build_parser().parse_args(["질문"])
        assert args.q_type == 3
        assert args.rerank_k == 5
        assert args.tokenizer_mode == "auto"

    def test_rejects_invalid_q_type(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["질문", "--q-type", "9"])

    def test_rejects_invalid_tokenizer_mode(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["질문", "--tokenizer-mode", "mecab"])


class TestExplainMode:
    def test_succeeds(self, corpus_file, capsys):
        code = main(["당뇨병의 합병증은?", "--corpus", str(corpus_file), "--explain"])
        assert code == 0
        assert "코퍼스: 3건" in capsys.readouterr().out

    def test_shows_tokenizer_used(self, corpus_file, capsys):
        """auto가 무엇으로 해석됐는지 출력에 남아야 재현이 가능합니다."""
        main(["당뇨병", "--corpus", str(corpus_file), "--explain"])
        assert "토크나이저:" in capsys.readouterr().out

    def test_shows_rrf_scores_and_contributions(self, corpus_file, capsys):
        main(["당뇨병의 합병증은?", "--corpus", str(corpus_file), "--explain"])
        out = capsys.readouterr().out
        assert "rrf=" in out
        assert "bm25" in out

    def test_show_prompt_flag(self, corpus_file, capsys):
        main(["당뇨병", "--corpus", str(corpus_file), "--explain", "--show-prompt"])
        out = capsys.readouterr().out
        assert "[참고자료]" in out
        assert "답변:" in out

    def test_no_results_still_exits_zero(self, corpus_file, capsys):
        """검색 결과가 없는 것은 오류가 아닙니다."""
        code = main(["zzzqqq", "--corpus", str(corpus_file), "--explain"])
        assert code == 0
        assert "후보 0건" in capsys.readouterr().out

    def test_tokenizer_mode_changes_results(self, corpus_file, capsys):
        main(["고혈압을 유지하려면", "--corpus", str(corpus_file),
              "--explain", "--tokenizer-mode", "whitespace"])
        ws = capsys.readouterr().out

        main(["고혈압을 유지하려면", "--corpus", str(corpus_file),
              "--explain", "--tokenizer-mode", "ngram"])
        ng = capsys.readouterr().out

        assert ws.count("rrf=") < ng.count("rrf=")


class TestErrorHandling:
    def test_missing_corpus_arg(self, capsys):
        code = main(["질문", "--explain"])
        assert code == 2
        assert "--corpus" in capsys.readouterr().err

    def test_without_explain_flag_refuses(self, capsys):
        """모델 없이 전체 파이프라인을 돌릴 수 없다고 안내합니다."""
        code = main(["질문"])
        assert code == 2
        assert "--explain" in capsys.readouterr().err

    def test_nonexistent_corpus(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["질문", "--corpus", str(tmp_path / "없음.jsonl"), "--explain"])

    def test_malformed_json_reports_line_number(self, tmp_path, capsys):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"text": "정상"}\n{깨진 줄}\n', encoding="utf-8")

        code = main(["질문", "--corpus", str(path), "--explain"])
        assert code == 1
        assert ":2" in capsys.readouterr().err

    def test_empty_corpus_file(self, tmp_path, capsys):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        code = main(["질문", "--corpus", str(path), "--explain"])
        assert code == 1
        assert "문서를 찾지 못했습니다" in capsys.readouterr().err

    def test_invalid_config_reports_cleanly(self, corpus_file, capsys):
        """설정 오류는 스택트레이스 대신 메시지로 알립니다."""
        code = main([
            "질문", "--corpus", str(corpus_file), "--explain",
            "--vector-k", "1", "--bm25-k", "1", "--rerank-k", "50",
        ])
        assert code == 2
        assert "설정 오류" in capsys.readouterr().err

    def test_blank_lines_skipped(self, tmp_path, capsys):
        path = tmp_path / "gaps.jsonl"
        path.write_text(
            '{"text": "당뇨병 개요", "file_id": "a"}\n\n\n'
            '{"text": "고혈압 개요", "file_id": "b"}\n',
            encoding="utf-8",
        )
        code = main(["당뇨병", "--corpus", str(path), "--explain"])
        assert code == 0
        assert "코퍼스: 2건" in capsys.readouterr().out


class TestDuplicateTolerance:
    def test_cli_keeps_first_on_duplicate(self, tmp_path, capsys):
        """CLI는 탐색 도구이므로 중복이 있어도 멈추지 않고 보고만 합니다."""
        path = tmp_path / "dup.jsonl"
        path.write_text(
            '{"text": "본문 하나", "file_id": "same", "chunk_index": 1}\n'
            '{"text": "본문 둘", "file_id": "same", "chunk_index": 1}\n',
            encoding="utf-8",
        )
        code = main(["본문", "--corpus", str(path), "--explain"])
        assert code == 0
        assert "중복 ID로 건너뛴 문서: 1건" in capsys.readouterr().out

    def test_cli_reports_collisions(self, tmp_path, capsys):
        """서로 다른 본문이 같은 ID로 합쳐지면 출력에 남아야 합니다."""
        path = tmp_path / "collide.jsonl"
        # chunk_index를 명시해 인덱싱은 통과시키되, 같은 doc_id가 되도록 만듭니다.
        path.write_text(
            '{"text": "본문 하나", "doc_id": "X", "chunk_index": 1}\n'
            '{"text": "본문 둘", "doc_id": "X", "chunk_index": 1}\n',
            encoding="utf-8",
        )
        code = main(["본문", "--corpus", str(path), "--explain"])
        assert code == 0
        # 인덱싱 단계에서 걸러지므로 중복 보고가 나옵니다.
        assert "중복 ID로 건너뛴 문서" in capsys.readouterr().out
