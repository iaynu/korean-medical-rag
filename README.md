# Korean Medical RAG

한국어 의료 QA를 위한 하이브리드 RAG 파이프라인입니다.
BM25와 벡터 검색을 **RRF로 융합**하고, CrossEncoder로 재랭킹한 뒤 LLM으로 답변을 생성합니다.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-231%20passed-brightgreen)](tests/)

## 한눈에 보기

| | |
|---|---|
| **무엇을 하나** | 한국어 의료 질문에 대해 BM25와 벡터 검색 결과를 RRF로 합치고, CrossEncoder로 재랭킹한 뒤 LLM에 근거 문서와 함께 넘깁니다 |
| **입력 / 출력** | 질문 + 문서 코퍼스(JSONL 또는 Chroma) → 근거 문서, 프롬프트, 답변, 단계별 중간 결과(`trace()`) |
| **원본 대비 수정** | 한국어 BM25 토크나이저, 순위를 버리던 후보 병합, 문서 ID 충돌 (아래 "원본에서 고친 것") |
| **코어 의존성** | `rank-bm25` 하나. 임베딩·재랭킹·생성 모델은 extras로 분리 |
| **검증** | 테스트 231건, 소규모 합성 벤치마크 (원본 코퍼스 재측정은 없음) |
| **GPU / 모델** | 검색·융합 단계 확인과 테스트에는 필요 없음 |

## 빠른 시작

모델도 GPU도 없이 검색·융합 단계를 바로 확인할 수 있습니다.

```bash
git clone https://github.com/iaynu/korean-medical-rag.git
cd korean-medical-rag
pip install -e ".[dev]"

krag "당뇨병의 합병증은 무엇인가?" --corpus examples/sample_corpus.jsonl --explain   # 검색·융합 결과
python examples/tokenizer_benchmark.py                                              # 토크나이저 비교
pytest                                                                              # 테스트 231건
```

## 목차

- [원본에서 고친 것](#원본에서-고친-것) · [측정](#측정)
- [설치](#설치) · [사용법](#사용법) · [구조](#구조)
- [테스트](#테스트) · [알려진 한계](#알려진-한계) · [확장하기](#확장하기) · [예외 레퍼런스](#예외-레퍼런스)
- [자주 묻는 질문](#자주-묻는-질문) · [검증 환경](#검증-환경) · [로드맵](#로드맵)

---

## 이 저장소에 대하여

원본은 **[Aegis-Bio-Sentinels](https://github.com/PhamPham2S/Aegis-Bio-Sentinels)**
(멋쟁이사자처럼 실전 프로젝트, 2026.01, 5인 팀 '뉴진스는5명')입니다.
**설계·구현·실험은 팀원 5명이 함께 진행했습니다.** 특정 모듈을 제가 단독으로
담당한 것이 아니므로, 원본의 어느 부분을 "제 것"이라고 나눌 수 없습니다.

이 저장소가 제 개인 작업인 부분은 **그 이후**입니다. 팀이 함께 만든 Colab
노트북을 읽고, 재현 가능한 파이썬 패키지로 재구성하면서 검색 단계의 문제
세 가지를 찾아 고쳤습니다. 아래 표가 그 경계입니다.

| | 원본 팀 프로젝트 | 이 저장소 |
|---|---|---|
| 주체 | 팀원 5인 공동 | 개인 작업 |
| 산출물 | Colab 노트북 29셀 | 9개 모듈 패키지 + 테스트 231건 |
| 성능 수치 (0.683 / 0.881 / 0.886) | 팀이 함께 측정 | 이 저장소에서 재측정하지 않음 |
| 검색 로직 | 원본 그대로 | 문제 3건 수정 (아래) |

원본의 실험 결과와 성능 지표는 팀의 성과입니다. 이 저장소는 그 수치를
다시 주장하지 않습니다. 여기서 제시하는 수치는 아래 "측정" 절의
소규모 재현 벤치마크뿐이며, 원본 코퍼스에서의 성능이 아닙니다.

전체 프로젝트 맥락은 [원본 저장소](https://github.com/PhamPham2S/Aegis-Bio-Sentinels)를 참고하세요.

---

## 프로젝트 배경

의료 질문에 LLM이 그대로 답하면 할루시네이션 위험이 큽니다. Fine-tuning은
새 지식 반영이 어렵고 긴 문맥 학습이 비효율적입니다. 그래서 검색으로 근거를
찾아 붙이는 RAG를 택했습니다.

```
질문
 │
 ├─→ 벡터 검색 (bge-m3)      k=20 ─┐
 │                                  ├─→ RRF 융합 ─→ 재랭킹 ─→ EXAONE-3.5 ─→ 답변
 └─→ BM25 검색               k=20 ─┘              (bge-reranker-v2-m3, k=5)
```

**데이터**: 의료 도메인 문서 약 55,918개 JSON, 평가셋 Q&A 약 3.5만 쌍
(객관식 731 / 단답형 116 / 서술형 153으로 구성된 1,000건 서브셋 기준).
데이터와 Vector DB는 이 저장소에 포함되어 있지 않습니다.

---

## 원본에서 고친 것

노트북을 옮기다 보니 검색 단계에 문제가 세 개 있었습니다. 셋 다 **조용히**
동작한다는 공통점이 있습니다. 에러가 나지 않고 결과가 그럴듯해서, 성능이
덜 나와도 원인으로 의심하기 어려운 종류입니다.

### 1. BM25가 한국어에서 사실상 동작하지 않았다

원본의 토크나이저입니다.

```python
def simple_tokenize(s):
    return s.replace("\n", " ").split()
```

한국어는 교착어라 어간에 조사가 붙습니다. 공백으로만 자르면 이렇게 됩니다.

```
질의: "당뇨병 합병증"      → {당뇨병, 합병증}
문서: "당뇨병의 합병증은"   → {당뇨병의, 합병증은}
교집합: ∅  →  BM25 점수 0
```

사람 눈에는 같은 내용인데 점수가 0입니다. 하이브리드 검색에서 BM25를 넣은
이유가 "의미는 비슷하나 표현이 다른" 벡터 검색의 약점을 **키워드 정확 매칭**으로
보완하는 것이었는데, 그 정확 매칭이 조사 때문에 깨지고 있었습니다.

**해결**: 한글 구간은 문자 n-gram으로, 라틴/숫자는 통째로 처리합니다.
`kiwipiepy`가 설치되어 있으면 형태소 분석을 씁니다
([`tokenization.py`](src/korean_medical_rag/tokenization.py)).

```
"당뇨병의"  → {당뇨, 뇨병, 병의}
"당뇨병"    → {당뇨, 뇨병}
교집합: {당뇨, 뇨병}  →  매칭 성공
```

조사를 직접 떼는 방식은 택하지 않았습니다. "사과"의 "과", "이빨"의 "이"처럼
조사와 형태가 같은 음절이 단어 안에 흔해서, 규칙 기반 제거는 멀쩡한 단어를
망가뜨립니다. n-gram은 그런 판단 자체를 하지 않아 더 안전합니다.

`PM2.5`, `COVID-19`, `HbA1c` 같은 의학 용어는 n-gram으로 쪼개지 않습니다.
이미 정확 매칭이 잘 되는 형태라 쪼개면 노이즈만 늘어납니다.

### 2. "융합"이 순위 정보를 버리고 있었다

원본의 `merge_candidates`는 이름과 달리 융합을 하지 않았습니다.

```python
merged = {}
for it in bm_items:                      # BM25 결과를 넣고
    merged[get_id(...)] = it
for it in vec_items:                     # 벡터 결과 중 없는 것만 추가
    if key not in merged:
        merged[key] = it
return list(merged.values())[:max_items] # 삽입 순서대로 자름
```

두 검색기가 매긴 **순위가 어디에도 반영되지 않습니다.** 양쪽 모두 1위로 꼽은
문서와 한쪽에서만 20위로 걸린 문서가 똑같은 자격으로 재랭커에 넘어갑니다.

점수를 그냥 더하지 못한 데는 이유가 있습니다. 벡터 점수는 거리를 뒤집은
값(대략 -2 ~ 0), BM25는 상한 없는 값(0 ~ 50+)이라 스케일이 다릅니다.
그대로 더하면 BM25가 언제나 이깁니다.

> **이 결함의 실제 영향 범위**
>
> 정직하게 말하면, 최종 평가에 쓰인 `hybrid_rerank` 모드에서는 이 문제가
> 결과를 크게 망치지 않았을 가능성이 큽니다. 후보가 20+20=40건이고
> `max_items=40`이라 잘림이 발생하지 않고, CrossEncoder가 40건을 전부
> 다시 채점하기 때문입니다. 순서가 나빠도 재랭커가 뒤에서 정리합니다.
>
> 문제가 실제로 드러나는 곳은 두 군데입니다. 첫째, 재랭커를 끄는
> `mode="hybrid"` 경로는 `merged[:5]`를 그대로 씁니다. 삽입 순서상
> **BM25 상위 5건**이 선택되고 벡터 검색 결과는 전혀 반영되지 않습니다.
> 하이브리드라는 이름이 무색해집니다. 둘째, 후보를 늘릴 때입니다.
> CrossEncoder는 후보당 forward pass를 한 번씩 돌리므로 k를 키우면
> 재랭킹 비용이 선형으로 늘어나는데, 순위가 없으면 어디서 잘라야
> 안전한지 알 수 없습니다.

**해결**: [RRF (Reciprocal Rank Fusion)](https://dl.acm.org/doi/10.1145/1571941.1572114)
— 점수를 보지 않고 **순위만** 씁니다.

```
score(d) = Σ_r  1 / (k + rank_r(d))
```

스케일에 영향받지 않고, 정규화 파라미터 튜닝이 필요 없으며, 검색기를 더
붙여도 그대로 동작합니다. 핵심 성질은 **합의가 단독 1위를 이긴다**는 것입니다.

```
벡터:  [A, C, D]      A는 벡터만의 1위
BM25:  [B, C, D]      B는 BM25만의 1위
                      C는 양쪽 모두의 2위

RRF 결과: C > A > B > D     ← C가 1위
```

두 검색기가 독립적으로 상위에 올렸다는 사실이, 한쪽에서만 1위인 것보다
강한 신호입니다. 융합 결과에는 어느 검색기가 몇 위로 올렸는지
(`contributions`)를 함께 남겨서, 검색 실패의 원인을 사후에 따질 수 있게
했습니다 ([`fusion.py`](src/korean_medical_rag/fusion.py)).

순위가 생기면 **후보를 안전하게 좁힐 수 있습니다.** `fusion_top_n`으로
재랭커에 넘길 후보 수를 제한하면, CrossEncoder forward pass가 그만큼
줄어듭니다. 기본값은 `None`(전부 넘김)이라 원본과 동일하게 동작하고,
비용을 줄이고 싶을 때만 켜면 됩니다.

```python
RagConfig(db_path="./db", vector_k=50, bm25_k=50, rerank_k=5, fusion_top_n=20)
# 후보 100건 → RRF 상위 20건만 재랭킹 → 5건 선정
```

### 3. 문서가 조용히 사라지고 있었다

```python
def get_id(text, meta):
    return m.get("file_id") or m.get("file_name") or text[:80]
```

메타데이터에 ID가 없으면 **본문 앞 80자**를 ID로 씁니다. 의료 지침 문서는
같은 원본을 청크로 나누면 도입부가 그대로 반복되는 경우가 많습니다.

```
청크 A: "본 지침은 대한당뇨병학회가 2023년에 발간한 제2형 당뇨병 진료 권고안으로, …제1장 진단 기준"
청크 B: "본 지침은 대한당뇨병학회가 2023년에 발간한 제2형 당뇨병 진료 권고안으로, …제2장 약물 치료"
                                    ↑ 앞 80자가 동일 → 같은 ID
```

융합 단계의 딕셔너리에서 한쪽이 조용히 덮어써집니다. 같은 `file_id`를 가진
서로 다른 청크도 마찬가지입니다.

**해결**: 메타데이터 ID에 청크 인덱스를 붙여 구분하고, ID가 없으면 본문
**전체의 SHA-1**을 씁니다 ([`retrieval.py`](src/korean_medical_rag/retrieval.py)).

```python
make_doc_id("청크 1", {"file_id": "med_001", "chunk_index": 0})  # 'med_001#0'
make_doc_id("본문")                                              # 'sha1:c67b8718...'
```

방어선은 두 겹입니다. `BM25Retriever`는 **인덱싱 시점**에 충돌을 감지하고
예외를 던집니다. 하지만 벡터 검색은 Chroma가 돌려준 메타데이터를 그대로
쓰기 때문에 그런 시점이 없습니다. 그래서 융합 단계에도 검사를 뒀습니다 —
같은 ID로 서로 다른 본문이 들어오면 `detect_id_collisions`가 세어서
`RagResult.id_collisions`로 보고합니다.

충돌이 있을 때의 출력 형태입니다 (예시).

```
질문: 당뇨병의 1차 약제는? (유형 2)
융합: 후보 34건 (양쪽 6건, vector 단독 14건, bm25 단독 14건)
재랭킹: 5건 선정
경고: ID 충돌 2건 — 서로 다른 본문이 같은 ID로 합쳐졌습니다 (med_014, med_031)
```

융합은 여전히 같은 ID를 하나로 합칩니다. 달라진 것은 **그 사실을 알 수
있다는 점**입니다. 원본에서는 문서가 사라져도 아무 신호가 없었습니다.

---

## 측정

토크나이저 변경의 효과를 확인하는 스크립트를 넣었습니다. GPU도 외부
데이터도 필요 없습니다.

```bash
python examples/tokenizer_benchmark.py
```

```
코퍼스 12건, 질의 10건, k=3

토크나이저                   Recall@3     MRR     결과 0건
----------------------------------------------------
공백 분리 (원본)                 0.700   0.700       3건
문자 n-gram (개선)             1.000   0.950       0건

Recall@3 0.700 → 1.000, 검색 결과 없음 3건 → 0건

질의별 결과:
  O → O  당뇨병의 합병증은 무엇인가?
  O → O  제2형 당뇨병에는 어떤 약을 쓰나?
  X → O  인슐린이 분비되는 곳은?
  O → O  고혈압의 진단 기준이 궁금하다
  X → O  고혈압을 치료하려면 어떻게 하나?
  O → O  심근경색이 의심될 때의 검사는?
  O → O  천식 발작에는 무엇을 쓰나?
  O → O  만성폐쇄성폐질환의 위험인자는?
  O → O  항생제는 바이러스에도 듣나?
  X → O  HbA1c는 무엇을 반영하나?

  O=정답 검색됨  X=결과 0건  -=결과는 있으나 정답 아님
```

> **이 수치를 어떻게 읽어야 하는가**
>
> 문서 12건, 질의 10건짜리 **합성 코퍼스**입니다. 조사가 붙은 질의에서
> 공백 분리가 무너진다는 것을 재현하려고 만든 예시이지, 원본 55,918건
> 코퍼스에서의 성능이 아닙니다. 원본 데이터에 재적용한 결과는 아직
> 없습니다 — 데이터 접근 권한과 GPU가 필요합니다.
>
> 표본이 작아 절대 수치의 의미는 제한적입니다. 다만 "결과 0건 3건 → 0건"은
> 스케일과 무관한 성질입니다. 질의 토큰이 코퍼스에 하나도 없으면 BM25는
> 코퍼스 크기와 상관없이 0건을 반환합니다.

---

## 설치

```bash
git clone https://github.com/iaynu/korean-medical-rag.git
cd korean-medical-rag
pip install -e ".[dev]"
```

코어 의존성은 `rank-bm25` 하나입니다. 임베딩·재랭킹·생성 모델은 extras로
분리했습니다. 검색·융합 로직만 확인할 때 torch를 받지 않아도 되게 하려는
의도입니다.

```bash
pip install -e ".[morph]"   # kiwipiepy 형태소 분석
pip install -e ".[models]"  # torch, transformers, sentence-transformers, chromadb
pip install -e ".[eval]"    # pandas, bert-score
```

---

## 사용법

### 검색 단계만 확인 (모델 불필요)

```bash
krag "당뇨병의 합병증은 무엇인가?" \
    --corpus examples/sample_corpus.jsonl --explain
```

```
코퍼스: 12건  토크나이저: 문자 n-gram
후보 2건 (양쪽 0건, bm25 단독 2건)

[1] rrf=0.01639 {'bm25': 1} 당뇨병의 주요 합병증으로는 당뇨병성 신증, 망막병증, 신경병증이 있으며 혈당 조절이 예방의 핵심이다.
[2] rrf=0.01613 {'bm25': 2} 제2형 당뇨병 환자에게는 메트포르민이 1차 약제로 권장된다. 신기능 저하 시에는 용량 조절이 필요하다.
```

`--show-prompt`를 붙이면 LLM에 실제로 들어가는 프롬프트 전문까지 출력합니다.

토크나이저를 바꿔가며 비교할 수 있습니다.

```bash
krag "고혈압을 치료하려면" --corpus examples/sample_corpus.jsonl \
     --explain --tokenizer-mode whitespace   # 결과 0건
krag "고혈압을 치료하려면" --corpus examples/sample_corpus.jsonl \
     --explain --tokenizer-mode ngram        # 결과 2건
```

### 파이썬 API

```python
from korean_medical_rag import RagConfig
from korean_medical_rag.pipeline import RagPipeline
from korean_medical_rag.retrieval import BM25Retriever, VectorRetriever
from korean_medical_rag.rerank import CrossEncoderReranker

config = RagConfig(db_path="./chroma_db", vector_k=20, bm25_k=20, rerank_k=5)

pipeline = RagPipeline(
    config=config,
    retrievers=[
        VectorRetriever(collection, embed_fn),
        BM25Retriever(texts, metadatas, tokenizer_mode="auto"),
    ],
    reranker=CrossEncoderReranker(),
    generator=my_generator,
    tokenizer=hf_tokenizer,
)

result = pipeline.answer("제2형 당뇨병의 1차 약제는?", q_type=2)
print(result.answer)
print(result.trace())
```

출력 예시 (실제 DB·GPU 환경에서의 형태를 보이기 위한 것으로, 이 저장소에서
실행한 결과가 아닙니다):

```
질문: 제2형 당뇨병의 1차 약제는? (유형 2)
융합: 후보 34건 (양쪽 6건, vector 단독 14건, bm25 단독 14건)
재랭킹: 5건 선정
소요: retrieve 82.4ms fuse 0.3ms rerank 141.2ms generate 1830.5ms
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--vector-k` | `20` | 벡터 검색 후보 수 |
| `--bm25-k` | `20` | BM25 검색 후보 수 |
| `--rerank-k` | `5` | 재랭킹 후 LLM에 넘길 문서 수 |
| `--rrf-k` | `60` | RRF 상수. 클수록 상위권 가중치가 완만 |
| `fusion_top_n` | `None` | 재랭커에 넘길 후보 상한 (파이썬 API) |
| `--tokenizer-mode` | `auto` | `auto` / `morph` / `ngram` / `whitespace` |
| `--q-type` | `3` | `1`=객관식 `2`=단답형 `3`=서술형 |
| `--explain` | `False` | 모델 없이 검색·융합만 실행 |

---

## 구조

```
src/korean_medical_rag/
├── config.py         # RagConfig — frozen dataclass, 생성 시점 검증
├── tokenization.py   # 한국어 토크나이저 (외부 의존성 없음)
├── fusion.py         # RRF 융합 (외부 의존성 없음)
├── retrieval.py      # 문서 ID, BM25Retriever, VectorRetriever
├── rerank.py         # CrossEncoder 재랭킹 (torch 지연 import)
├── prompts.py        # q_type별 프롬프트
├── pipeline.py       # 오케스트레이션
├── exceptions.py     # 예외 계층
└── cli.py            # 커맨드라인 진입점
```

의존성 방향은 단방향입니다.

```
cli ──→ pipeline ──→ {retrieval, prompts} ──→ {config, fusion, tokenization, exceptions}
 └──────────────────────────┴────────────────────→ (cli도 config·fusion·retrieval 직접 사용)

rerank ──→ {config, fusion}          # pipeline은 rerank를 import하지 않음
```

`pipeline.py`는 `rerank`를 **import하지 않습니다.** 재랭커를 `Protocol`로만
선언하고 객체를 주입받습니다. 이것이 모델 계층 격리의 실체입니다 — 파이프라인을
import해도 torch가 딸려오지 않습니다.

`--explain` 모드도 같은 `RagPipeline`을 씁니다. 재랭커와 생성기만 빼고
조립하므로, 여기서 본 검색·융합 결과가 전체 실행에서도 동일합니다.

**모델 계층을 격리했습니다.** `torch`와 `sentence-transformers` import는
`rerank.py`의 메서드 안에만 있습니다. 덕분에 검색·융합·프롬프트 로직을
GPU 없이 수 초 만에 테스트할 수 있습니다. 이 프로젝트에서 실제로 시간을
잡아먹은 문제는 대부분 모델이 아니라 데이터와 검색 로직 쪽이었고,
그걸 확인하려고 매번 7.8B 모델을 올릴 이유는 없었습니다.

**검색기는 프로토콜입니다.** 파이프라인은 `name`과 `retrieve(query, k)`만
알면 됩니다. 새 검색기를 붙일 때 파이프라인을 고칠 필요가 없고, 테스트에서는
가짜 검색기를 끼워 넣습니다.

---

## 테스트

```bash
pytest
```

```
231 passed
```

무거운 의존성 없이 돌아갑니다. 커버리지는 원본에서 문제가 났던 지점에
집중되어 있습니다.

- **조사 문제 재현** — 공백 분리로는 매칭이 안 되고 n-gram으로는 된다는 것을
  질의·문서 쌍으로 검증
- **원본 ID 충돌 재현** — `text[:80]` 방식이 실제로 충돌하는 조건을 테스트로
  고정. 새 구현이 그 조건에서 충돌하지 않음을 확인
- **RRF 스케일 불변성** — BM25 점수를 1000배로 키워도 순위가 동일. 대조군으로
  "점수를 그냥 더하면 BM25가 지배한다"는 것도 함께 검증
- **중복 반환 방어** — 한 검색기가 같은 문서를 3번 반환해도 점수가 부풀지 않음
- **동점 결정성** — 실행마다 순서가 달라지지 않아야 결과 비교가 가능
- **빈 결과 내성** — 검색 결과가 없어도 배치가 멈추지 않음
- **q_type 범위** — 원본이 `else`로 흘려보내던 잘못된 값을 거부
- **의학 용어 보존** — `PM2.5`, `COVID-19`, `HbA1c`가 쪼개지지 않음
- **모델 계층 격리** — torch를 쓰지 않아야 할 모듈 전부를 별도 프로세스에서 import한 뒤
  `sys.modules`에 torch·transformers가 없음을 확인. 리팩토링 중 이 성질이
  조용히 깨지는 것을 막습니다

---

## 알려진 한계

- **원본 코퍼스 재측정 없음.** 개선의 효과는 소규모 합성 벤치마크와 단위
  테스트로만 확인했습니다. 55,918건 코퍼스에서의 end-to-end 성능 변화는
  측정하지 않았습니다. 세 가지 수정이 최종 지표를 얼마나 움직이는지는
  **모릅니다.** 특히 결함 2는 위에 적었듯 `hybrid_rerank` 경로에서
  영향이 작을 수 있습니다.
- **ID 충돌은 벡터 경로에서 감지만 하고 막지는 못합니다.** Chroma가 돌려준
  결과에 대해서는 이미 색인이 끝난 뒤이므로, 융합 단계에서 보고할 수는
  있어도 원본 문서를 되살릴 수는 없습니다. 근본 해결은 색인 시점에
  `chunk_index`를 넣는 것이고, 그건 이 저장소 밖의 작업입니다.
- **BM25 인덱스가 메모리에 상주.** 원본과 동일하게 매 실행 시 코퍼스 전체를
  덤프해 인덱스를 만듭니다. 코퍼스가 커지면 영속화가 필요합니다.
- **RRF 상수 k와 `fusion_top_n`을 튜닝하지 않았습니다.** k는 원 논문의 60을
  그대로 쓰고, `fusion_top_n`은 기본값이 `None`이라 켜지 않으면 원본과
  동일하게 전 후보를 재랭킹합니다. 어디서 잘라야 안전한지는 실제 데이터로
  확인해야 합니다.
- **형태소 분석 경로의 테스트가 얕습니다.** `kiwipiepy`가 선택적 의존성이라
  CI에서는 n-gram 경로만 검증됩니다.
- **`CrossEncoderReranker`의 실제 추론은 테스트되지 않습니다.** 모델 로딩이
  일어나지 않는다는 것(지연 import, 빈 입력 조기 반환)까지만 검증합니다.

## 확장하기

파이프라인은 세 개의 `Protocol`만 알면 됩니다. 모델 없이도 끝까지 돌려 볼 수 있습니다.

| 부품 | 인터페이스 | 비고 |
|---|---|---|
| 검색기 | `name: str`, `retrieve(query, k) -> List[RetrievedDoc]` | `name`은 검색기끼리 겹치면 안 됩니다 |
| 재랭커 | `rerank(query, docs, top_k) -> List[FusedDoc]` | 생략하면 RRF 상위를 그대로 씁니다 |
| 생성기 | `generate(prompt) -> str` | 생략하면 프롬프트만 만들고 답변은 빈 문자열입니다 |

BM25 검색기와 가짜 생성기만으로 질문부터 trace까지 돌리는 예시입니다. 실제로 실행해서 확인한 코드입니다.

```python
import json
from korean_medical_rag import RagConfig
from korean_medical_rag.pipeline import RagPipeline
from korean_medical_rag.retrieval import BM25Retriever

rows = [json.loads(l) for l in open("examples/sample_corpus.jsonl", encoding="utf-8")]
bm25 = BM25Retriever([r["text"] for r in rows], [{"file_id": r["file_id"]} for r in rows])

class EchoGenerator:
    def generate(self, prompt: str) -> str:
        return f"(프롬프트 {len(prompt)}자)"

pipe = RagPipeline(config=RagConfig(db_path="./unused"), retrievers=[bm25], generator=EchoGenerator())
result = pipe.answer("당뇨병의 합병증은 무엇인가?", q_type=3)
print(result.trace())
```

```
질문: 당뇨병의 합병증은 무엇인가? (유형 3)
융합: 후보 2건 (양쪽 0건, bm25 단독 2건)
재랭킹: 2건 선정
소요: retrieve 0.2ms fuse 0.0ms rerank 0.0ms generate 0.0ms
```

새 검색기(예: 다른 임베딩 모델, 키워드 사전)를 붙일 때는 `retrieve`가 **순위대로** 정렬된 목록을 돌려주기만 하면 됩니다. RRF는 점수를 보지 않으므로 점수 단위를 맞출 필요가 없습니다.

답변이 틀렸을 때는 `RagResult`의 중간 결과로 원인 단계를 가립니다.

| 정답 문서의 위치 | 의심할 단계 |
|---|---|
| `fused`에도 없음 | 검색기 (토크나이저, 임베딩, `*_k`) |
| `fused`에는 있는데 `reranked`에서 빠짐 | 재랭커, `rerank_k`, `fusion_top_n` |
| `reranked`에 있는데 답이 틀림 | 프롬프트, 생성 모델 |

## 예외 레퍼런스

모든 예외는 `RagError`를 상속하므로 `except RagError`로 한 번에 받을 수 있습니다.

| 예외 | 언제 발생하나 |
|---|---|
| `ConfigError` | `RagConfig` 값이 잘못됨. 생성 시점에 바로 발생하므로 모델을 올리기 전에 드러납니다 |
| `EmptyCorpusError` | 코퍼스가 비어 있음 |
| `DuplicateDocumentIdError` | 인덱싱 시점에 같은 ID로 서로 다른 본문이 들어옴 |
| `CorpusError` | 코퍼스 형식 오류 (위 두 예외의 부모) |
| `RetrievalError` | 검색 단계 실패 |
| `GenerationError` | 생성 단계 실패 |

## 자주 묻는 질문

**`--explain` 결과가 몇 건 안 나옵니다.**
BM25는 질의 토큰이 문서에 하나라도 있어야 후보로 올립니다. 합성 코퍼스는 12건뿐이라 후보가 2건 정도로 적게 나오는 것이 정상입니다. 벡터 검색기를 붙이면 후보가 늘어납니다.

**`kiwipiepy`를 설치하지 않아도 되나요?**
됩니다. `--tokenizer-mode auto`는 설치돼 있으면 형태소 분석을, 없으면 문자 n-gram을 씁니다. 어느 쪽이 쓰였는지는 `--explain` 첫 줄의 "토크나이저"에 나옵니다. 형태소 모드를 명시했는데 `kiwipiepy`가 없으면 자동 폴백 없이 `ImportError`가 납니다.

**원본과 같은 동작(공백 분리)으로 돌려 보고 싶습니다.**
`--tokenizer-mode whitespace`로 지정하면 됩니다. 조사가 붙은 질의에서 결과가 0건이 되는 것을 직접 비교해 볼 수 있습니다.

**`id_collisions` 경고가 뜹니다.**
서로 다른 본문이 같은 ID로 합쳐졌다는 뜻입니다. 문서가 하나 사라진 상태이므로 색인 시점에 `chunk_index` 같은 값을 메타데이터에 넣어 다시 색인해야 합니다. 이 저장소는 감지까지만 합니다 ("알려진 한계" 참고).

**원본 코퍼스에서 성능이 얼마나 오르나요?**
측정하지 않았습니다. 이 저장소가 보이는 수치는 합성 벤치마크가 전부입니다.

## 검증 환경

이 README의 명령과 출력은 아래 환경에서 직접 실행해 확인했습니다.

| | |
|---|---|
| OS / Python | Windows 11, Python 3.12.10 |
| 설치 | `pip install -e ".[dev]"` (모델 extras 없이) |
| 결과 | `pytest` 231건 통과, `krag --explain`과 `tokenizer_benchmark.py` 출력이 위 문서와 일치 |

`models` extras를 설치한 상태에서의 임베딩·재랭킹·생성 실행은 확인하지 않았습니다. 위 "파이썬 API" 절의 출력 예시가 그 때문에 예시라고 명시되어 있습니다.

## 로드맵

"알려진 한계"에서 해결 가능한 항목을 작업 단위로 옮긴 목록입니다. 아직 시작하지 않았습니다.

- [ ] 원본 코퍼스에서 세 가지 수정의 end-to-end 효과 측정 (데이터 접근과 GPU 필요)
- [ ] `fusion_top_n`과 RRF 상수 `k` 튜닝
- [ ] BM25 인덱스 영속화
- [ ] `kiwipiepy` 경로를 CI에서 검증 (선택적 의존성을 포함한 별도 잡)
- [ ] GitHub Actions로 테스트를 자동 실행하고 테스트 배지를 정적 이미지에서 실제 상태 배지로 교체

---

## 라이선스

MIT — [LICENSE](LICENSE) 참고.

원본 학습·평가 데이터는 별도 이용 약관을 따르며 이 저장소에 포함되어 있지
않습니다. `examples/sample_corpus.jsonl`은 동작 확인용으로 직접 작성한
합성 데이터입니다.
