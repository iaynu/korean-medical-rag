"""토크나이저가 BM25 검색 결과에 미치는 영향을 측정합니다.

README의 주장을 독자가 직접 확인할 수 있도록 하려는 스크립트입니다.
실행에 GPU나 외부 데이터가 필요 없습니다.

    python examples/tokenizer_benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from korean_medical_rag.retrieval import BM25Retriever

CORPUS_PATH = Path(__file__).parent / "sample_corpus.jsonl"

# (질의, 정답으로 간주할 file_id) — 조사가 붙은 자연스러운 질문들입니다.
# 사람이 보기에는 코퍼스의 특정 문서를 명백히 가리킵니다.
QUERIES: List[Tuple[str, str]] = [
    ("당뇨병의 합병증은 무엇인가?", "med_000"),
    ("제2형 당뇨병에는 어떤 약을 쓰나?", "med_001"),
    ("인슐린이 분비되는 곳은?", "med_002"),
    ("고혈압의 진단 기준이 궁금하다", "med_003"),
    ("고혈압을 치료하려면 어떻게 하나?", "med_004"),
    ("심근경색이 의심될 때의 검사는?", "med_005"),
    ("천식 발작에는 무엇을 쓰나?", "med_006"),
    ("만성폐쇄성폐질환의 위험인자는?", "med_007"),
    ("항생제는 바이러스에도 듣나?", "med_010"),
    ("HbA1c는 무엇을 반영하나?", "med_011"),
]

MODES = ["whitespace", "ngram"]
MODE_LABELS = {
    "whitespace": "공백 분리 (원본)",
    "ngram": "문자 n-gram (개선)",
}


def load_corpus() -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metas.append({k: v for k, v in record.items() if k != "text"})
    return texts, metas


def evaluate(mode: str, texts, metas, k: int = 3) -> dict:
    """Recall@k와 MRR을 계산합니다.

    Recall@k: 정답 문서가 상위 k건 안에 있는 질의의 비율.
    MRR: 정답 문서 순위의 역수 평균. 못 찾으면 0.
    """
    retriever = BM25Retriever(texts, metas, tokenizer_mode=mode)

    hits = 0
    reciprocal_ranks = []
    empty = 0

    for query, gold_id in QUERIES:
        results = retriever.retrieve(query, k=k)
        if not results:
            empty += 1
            reciprocal_ranks.append(0.0)
            continue

        found_ids = [d.doc_id for d in results]
        if gold_id in found_ids:
            hits += 1
            reciprocal_ranks.append(1.0 / (found_ids.index(gold_id) + 1))
        else:
            reciprocal_ranks.append(0.0)

    n = len(QUERIES)
    return {
        "recall@k": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "empty": empty,
    }


def main() -> None:
    texts, metas = load_corpus()
    k = 3

    print(f"코퍼스 {len(texts)}건, 질의 {len(QUERIES)}건, k={k}")
    print()
    print(f"{'토크나이저':<22} {'Recall@3':>9} {'MRR':>7} {'결과 0건':>9}")
    print("-" * 52)

    results = {}
    for mode in MODES:
        m = evaluate(mode, texts, metas, k=k)
        results[mode] = m
        print(
            f"{MODE_LABELS[mode]:<22} "
            f"{m['recall@k']:>9.3f} {m['mrr']:>7.3f} "
            f"{m['empty']:>7}건"
        )

    print()
    base, improved = results["whitespace"], results["ngram"]
    print(
        f"Recall@3 {base['recall@k']:.3f} → {improved['recall@k']:.3f}, "
        f"검색 결과 없음 {base['empty']}건 → {improved['empty']}건"
    )
    print()
    print("질의별 결과:")
    for query, gold_id in QUERIES:
        row = []
        for mode in MODES:
            r = BM25Retriever(texts, metas, tokenizer_mode=mode).retrieve(query, k=k)
            found = gold_id in [d.doc_id for d in r]
            row.append("O" if found else ("-" if r else "X"))
        print(f"  {row[0]} → {row[1]}  {query}")
    print()
    print("  O=정답 검색됨  X=결과 0건  -=결과는 있으나 정답 아님")


if __name__ == "__main__":
    main()
