"""커맨드라인 진입점.

``--explain`` 은 모델 없이 검색·융합 단계만 돌립니다. 검색이 정답 문서를
못 찾는 것인지, 찾았는데 LLM이 못 쓰는 것인지 구분하는 용도입니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from .config import RagConfig
from .exceptions import RagError
from .fusion import fusion_summary
from .pipeline import RagPipeline
from .retrieval import BM25Retriever
from .tokenization import describe_mode

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krag",
        description="한국어 의료 QA 하이브리드 RAG 파이프라인",
    )
    parser.add_argument("question", help="질문 문장")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="JSONL 코퍼스 경로. 각 줄에 text 필드 필요. "
        "--explain 모드에서 Chroma 없이 BM25만 돌릴 때 씁니다.",
    )
    parser.add_argument("--db-path", default="", help="Chroma 영속 디렉터리")
    parser.add_argument("--collection", default="med_knowledge")
    parser.add_argument("--q-type", type=int, default=3, choices=[1, 2, 3],
                        help="1=객관식 2=단답형 3=서술형")
    parser.add_argument("--vector-k", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--tokenizer-mode", default="auto",
                        choices=["auto", "morph", "ngram", "whitespace"])
    parser.add_argument("--explain", action="store_true",
                        help="모델을 올리지 않고 검색·융합 결과와 프롬프트만 출력")
    parser.add_argument("--show-prompt", action="store_true",
                        help="완성된 프롬프트를 출력")
    return parser


def _load_jsonl_corpus(path: Path) -> tuple[List[str], List[dict]]:
    texts: List[str] = []
    metas: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RagError(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc
            text = record.get("text") or record.get("question") or ""
            if not text:
                continue
            texts.append(text)
            meta = {k: v for k, v in record.items() if k != "text"}
            meta.setdefault("chunk_index", lineno)
            metas.append(meta)
    return texts, metas


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = RagConfig(
            db_path=args.db_path or "(unset)",
            collection_name=args.collection,
            vector_k=args.vector_k,
            bm25_k=args.bm25_k,
            rerank_k=args.rerank_k,
            rrf_k=args.rrf_k,
            tokenizer_mode=args.tokenizer_mode,
        )
    except RagError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2

    if not args.explain:
        print(
            "전체 파이프라인 실행에는 Chroma DB와 GPU가 필요합니다.\n"
            "먼저 --explain 으로 검색 단계만 확인해 보세요.",
            file=sys.stderr,
        )
        return 2

    if not args.corpus:
        print("--explain 모드에는 --corpus 가 필요합니다", file=sys.stderr)
        return 2

    try:
        texts, metas = _load_jsonl_corpus(args.corpus)
        if not texts:
            print(f"{args.corpus}에서 문서를 찾지 못했습니다", file=sys.stderr)
            return 1

        retriever = BM25Retriever(
            texts, metas,
            tokenizer_mode=config.tokenizer_mode,
            on_duplicate="keep_first",
        )

        # 재랭커도 생성기도 없이 파이프라인을 조립합니다. 검색 → 융합 →
        # 프롬프트까지는 전체 실행과 동일한 코드 경로를 지나므로, 여기서
        # 본 결과가 실제 실행에서도 같습니다.
        pipeline = RagPipeline(config=config, retrievers=[retriever])
        result = pipeline.answer(args.question, args.q_type)

        print(f"코퍼스: {len(retriever)}건  토크나이저: {describe_mode(config.tokenizer_mode)}")
        if retriever.duplicate_count:
            print(f"중복 ID로 건너뛴 문서: {retriever.duplicate_count}건")
        if result.id_collisions:
            print(f"ID 충돌: {len(result.id_collisions)}건")
        print(fusion_summary(result.fused))
        print()

        for i, doc in enumerate(result.reranked, start=1):
            preview = doc.text.replace("\n", " ")[:70]
            print(f"[{i}] rrf={doc.rrf_score:.5f} {dict(doc.contributions)} {preview}")

        if args.show_prompt:
            print()
            print("=" * 60)
            print(result.prompt)

    except RagError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
