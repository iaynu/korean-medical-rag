"""모델 계층 격리 검증.

README는 "torch 없이 검색·융합·프롬프트를 테스트할 수 있다"고 주장합니다.
그 주장이 리팩토링 중에 조용히 깨지는 것을 막는 테스트입니다. 누군가
config.py 상단에 torch를 import하면 여기서 잡힙니다.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# torch를 절대 끌어오면 안 되는 모듈들.
PURE_MODULES = [
    "korean_medical_rag.config",
    "korean_medical_rag.exceptions",
    "korean_medical_rag.tokenization",
    "korean_medical_rag.fusion",
    "korean_medical_rag.prompts",
    "korean_medical_rag.retrieval",
    "korean_medical_rag.pipeline",
    "korean_medical_rag.cli",
    "korean_medical_rag",
]

HEAVY = ["torch", "transformers", "sentence_transformers", "chromadb"]


@pytest.mark.parametrize("module", PURE_MODULES)
def test_module_does_not_import_heavy_deps(module):
    """별도 프로세스에서 import한 뒤 sys.modules를 확인합니다."""
    code = (
        f"import {module}; import sys; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    leaked = proc.stdout.strip()
    assert not leaked, f"{module}이 무거운 의존성을 끌어옵니다: {leaked}"


def test_rerank_module_import_is_also_lazy():
    """rerank 모듈 자체를 import해도 모델 라이브러리는 로드되지 않습니다.

    import는 TYPE_CHECKING 가드 뒤에 있고 실제 로딩은 메서드 안에서만
    일어나야 합니다.
    """
    code = (
        "import korean_medical_rag.rerank as r; import sys; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert not proc.stdout.strip(), f"rerank import가 무겁습니다: {proc.stdout}"


def test_reranker_construction_does_not_load_model():
    """생성자에서 모델을 올리면 안 됩니다. 조립만 하고 안 돌리는 경우가 있습니다."""
    from korean_medical_rag.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker(device="cpu")
    assert reranker._model is None

    # 후보가 없으면 모델을 올리지 않고 바로 반환합니다.
    assert reranker.rerank("질문", [], top_k=5) == []
    assert reranker._model is None


def test_reranker_rejects_invalid_args():
    from korean_medical_rag.rerank import CrossEncoderReranker

    with pytest.raises(ValueError, match="batch_size는 1 이상"):
        CrossEncoderReranker(batch_size=0)

    with pytest.raises(ValueError, match="top_k는 1 이상"):
        CrossEncoderReranker(device="cpu").rerank("질문", [], top_k=0)
