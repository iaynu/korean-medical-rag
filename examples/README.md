# 예제

## sample_corpus.jsonl

동작 확인용 **합성** 의료 문서 12건입니다. 원본 프로젝트의 학습 데이터는
재배포할 수 없어 직접 작성했습니다. 각 줄은 `text`, `file_id`, `domain`
필드를 가진 JSON 객체입니다.

조사가 붙은 표현이 섞이도록 구성했습니다. 토크나이저 차이가 드러나야
벤치마크가 의미를 갖기 때문입니다.

## tokenizer_benchmark.py

공백 분리(원본)와 문자 n-gram(개선)의 Recall@3 · MRR을 비교합니다.

```bash
python examples/tokenizer_benchmark.py
```

GPU도 외부 데이터도 필요 없습니다. 표본이 작으므로 절대 수치보다는
"조사 붙은 질의에서 공백 분리가 결과 0건을 내는가"를 보는 용도입니다.
