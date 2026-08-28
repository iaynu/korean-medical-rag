"""BM25용 한국어 토크나이저.

원본 노트북은 ``text.split()`` 으로 토큰을 만들었습니다. 영어라면 대체로
괜찮지만 한국어에서는 BM25가 사실상 동작하지 않습니다. 한국어는 교착어라
어간에 조사가 붙어 표면형이 달라지기 때문입니다.

    질의: "당뇨병 합병증"     → ["당뇨병", "합병증"]
    문서: "당뇨병의 합병증은"  → ["당뇨병의", "합병증은"]

두 토큰 집합의 교집합은 공집합입니다. 사람 눈에는 같은 내용인데 BM25 점수는
0입니다. 하이브리드 검색에서 BM25를 넣은 이유가 "의미는 비슷하지만 표현이
다른" 벡터 검색의 약점을 키워드 정확 매칭으로 보완하는 것이었는데, 정작
그 정확 매칭이 조사 때문에 깨지고 있었습니다.

해법은 두 가지입니다.

**형태소 분석** (``morph``) — ``kiwipiepy`` 가 설치되어 있으면 사용합니다.
어간을 정확히 뽑아내므로 품질이 가장 좋습니다. 다만 설치가 필요합니다.

**문자 n-gram** (``ngram``) — 의존성이 없습니다. "당뇨병의"의 bigram은
{당뇨, 뇨병, 병의}, "당뇨병"은 {당뇨, 뇨병}이라 두 개가 겹칩니다. 조사가
만드는 마지막 한두 개 bigram만 어긋나고 나머지는 그대로 매칭됩니다.
한국어 정보검색에서 오래 쓰인 견고한 기법입니다.

조사를 직접 떼는 방식은 택하지 않았습니다. "사과"의 "과", "이빨"의 "이"처럼
조사와 형태가 같은 음절이 단어 안에 흔해서, 규칙 기반 제거는 멀쩡한 단어를
망가뜨립니다. n-gram은 그런 판단 자체를 하지 않으므로 더 안전합니다.

라틴 문자와 숫자는 n-gram으로 쪼개지 않고 통째로 둡니다. 의학 문서에는
"PM2.5", "WHO", "COVID-19" 같은 토큰이 많은데, 이들은 이미 정확 매칭이
잘 되는 형태라 쪼개면 오히려 노이즈만 늘어납니다.
"""

from __future__ import annotations

import re
from typing import Callable, List

__all__ = [
    "DEFAULT_NGRAM_SIZE",
    "char_ngrams",
    "script_runs",
    "ngram_tokenize",
    "whitespace_tokenize",
    "morph_tokenize",
    "kiwi_available",
    "get_tokenizer",
    "describe_mode",
]

DEFAULT_NGRAM_SIZE = 2

# 한글 음절(가-힣), 한글 자모, 한자를 "한국어 문자"로 봅니다.
# 한자는 의학 용어에 병기되는 경우가 있어 포함합니다.
_HANGUL = r"가-힣ᄀ-ᇿ㄰-㆏一-鿿"
_LATIN_NUM = r"a-zA-Z0-9"

_RUN_PATTERN = re.compile(f"[{_HANGUL}]+|[{_LATIN_NUM}]+(?:[.\\-][{_LATIN_NUM}]+)*")
_IS_KOREAN = re.compile(f"^[{_HANGUL}]+$")


def script_runs(text: str) -> List[str]:
    """문자 종류가 같은 구간끼리 끊어냅니다.

    구두점과 공백은 버립니다. 라틴 문자열 안의 점과 하이픈은 살립니다.
    "PM2.5"나 "COVID-19"가 쪼개지면 안 되기 때문입니다.

    >>> script_runs("당뇨병(diabetes) 환자")
    ['당뇨병', 'diabetes', '환자']
    >>> script_runs("PM2.5 농도가 높다")
    ['PM2.5', '농도가', '높다']
    >>> script_runs("COVID-19 백신")
    ['COVID-19', '백신']
    >>> script_runs("혈압이120/80이다")
    ['혈압이', '120', '80', '이다']
    """
    if not isinstance(text, str):
        return []
    return _RUN_PATTERN.findall(text)


def char_ngrams(token: str, n: int = DEFAULT_NGRAM_SIZE) -> List[str]:
    """문자 n-gram을 만듭니다.

    토큰이 n보다 짧으면 토큰 자체를 하나 반환합니다. 버리면 "위", "간"
    같은 한 글자 장기 이름이 통째로 사라집니다.

    >>> char_ngrams("당뇨병")
    ['당뇨', '뇨병']
    >>> char_ngrams("당뇨병의")
    ['당뇨', '뇨병', '병의']
    >>> char_ngrams("간")
    ['간']
    >>> char_ngrams("합병증", n=3)
    ['합병증']
    """
    if n < 1:
        raise ValueError(f"n은 1 이상이어야 합니다 (받은 값: {n})")
    if len(token) <= n:
        return [token] if token else []
    return [token[i : i + n] for i in range(len(token) - n + 1)]


def whitespace_tokenize(text: str) -> List[str]:
    """원본 노트북이 쓰던 방식. 비교·회귀 테스트용으로 남겨둡니다.

    >>> whitespace_tokenize("당뇨병의 합병증은")
    ['당뇨병의', '합병증은']
    """
    if not isinstance(text, str):
        return []
    return text.replace("\n", " ").split()


def ngram_tokenize(text: str, n: int = DEFAULT_NGRAM_SIZE) -> List[str]:
    """한글은 n-gram으로, 라틴/숫자는 통째로 소문자화해서 반환합니다.

    >>> ngram_tokenize("당뇨병의 합병증")
    ['당뇨', '뇨병', '병의', '합병', '병증']
    >>> ngram_tokenize("WHO 기준")
    ['who', '기준']

    조사가 달라도 대부분의 n-gram이 겹칩니다. 이것이 이 모듈의 존재 이유입니다.

    >>> q = set(ngram_tokenize("당뇨병 합병증"))
    >>> d = set(ngram_tokenize("당뇨병의 합병증은"))
    >>> sorted(q & d)
    ['뇨병', '당뇨', '병증', '합병']

    같은 문장을 공백 분리로 처리하면 겹치는 토큰이 없습니다.

    >>> set(whitespace_tokenize("당뇨병 합병증")) & set(whitespace_tokenize("당뇨병의 합병증은"))
    set()
    """
    tokens: List[str] = []
    for run in script_runs(text):
        if _IS_KOREAN.match(run):
            tokens.extend(char_ngrams(run, n))
        else:
            tokens.append(run.lower())
    return tokens


def kiwi_available() -> bool:
    """kiwipiepy를 import할 수 있는지 확인합니다."""
    try:
        import kiwipiepy  # noqa: F401
    except ImportError:
        return False
    return True


# 분석기 인스턴스는 만드는 비용이 커서 한 번만 만들고 재사용합니다.
_kiwi = None

# 실질적 의미를 담는 품사만 남깁니다. 조사(J*), 어미(E*), 접미사(X*)는
# 검색어로서 변별력이 없고 거의 모든 문서에 나타나 점수를 왜곡합니다.
_CONTENT_POS = ("NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "MAG", "SL", "SN", "SH")


def morph_tokenize(text: str) -> List[str]:
    """kiwipiepy 형태소 분석으로 내용어 어간만 뽑습니다.

    kiwipiepy가 없으면 ImportError를 던집니다. 자동 폴백은
    :func:`get_tokenizer` 가 담당합니다.
    """
    global _kiwi
    if not isinstance(text, str) or not text.strip():
        return []
    if _kiwi is None:
        from kiwipiepy import Kiwi

        _kiwi = Kiwi()
    return [
        token.form.lower()
        for token in _kiwi.tokenize(text)
        if token.tag in _CONTENT_POS
    ]


def get_tokenizer(mode: str = "auto", n: int = DEFAULT_NGRAM_SIZE) -> Callable[[str], List[str]]:
    """모드 이름으로 토크나이저 함수를 얻습니다.

    ``auto`` 는 kiwipiepy가 있으면 형태소 분석을, 없으면 n-gram을 씁니다.
    설치 여부와 무관하게 항상 동작하되, 가능하면 더 좋은 쪽을 쓰기 위함입니다.

    >>> tok = get_tokenizer("ngram")
    >>> tok("당뇨병")
    ['당뇨', '뇨병']
    >>> get_tokenizer("whitespace")("당뇨병의 합병증")
    ['당뇨병의', '합병증']
    >>> get_tokenizer("없는모드")
    Traceback (most recent call last):
        ...
    ValueError: 알 수 없는 tokenizer_mode: '없는모드'
    """
    if mode == "auto":
        mode = "morph" if kiwi_available() else "ngram"

    if mode == "morph":
        return morph_tokenize
    if mode == "ngram":
        return lambda text: ngram_tokenize(text, n)
    if mode == "whitespace":
        return whitespace_tokenize
    raise ValueError(f"알 수 없는 tokenizer_mode: {mode!r}")


def describe_mode(mode: str) -> str:
    """실제로 선택된 토크나이저를 사람이 읽을 문자열로 설명합니다.

    ``auto`` 가 무엇으로 해석됐는지 로그에 남기기 위한 용도입니다.

    >>> describe_mode("ngram")
    '문자 n-gram'
    """
    if mode == "auto":
        mode = "morph" if kiwi_available() else "ngram"
    return {
        "morph": "형태소 분석 (kiwipiepy)",
        "ngram": "문자 n-gram",
        "whitespace": "공백 분리 (원본 방식)",
    }[mode]
