import re

import pytest

from sentinel_rosetta.application.use_cases import TranslateTextUseCase
from sentinel_rosetta.domain.models import LanguageMap
from sentinel_rosetta.infrastructure.ctranslate2_adapter import (
    CTranslate2Adapter,
)
from sentinel_rosetta.infrastructure.opencc_adapter import OpenCcAdapter
from tests.example_sentences import EXAMPLE_CHINESE, EXAMPLE_ENGLISH
from tests.model_environment import MODELS_DIR, missing_translation_models

_MISSING = missing_translation_models()

pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"translation models missing: {_MISSING}"
)


@pytest.fixture(scope="module")
def use_case() -> TranslateTextUseCase:
    # Loading the real models is costly and execute() keeps no state, so the
    # module's tests share one instance.
    return TranslateTextUseCase(
        CTranslate2Adapter(models_dir=MODELS_DIR), OpenCcAdapter()
    )


def test_execute_real_adapters_en_to_zh_returns_traditional_text(
    use_case: TranslateTextUseCase,
) -> None:
    # Act
    result = use_case.execute(
        EXAMPLE_ENGLISH,
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
    )

    # Assert
    assert result.target_text != ""
    assert OpenCcAdapter().to_traditional(result.target_text) == (
        result.target_text
    )


def test_execute_real_adapters_zh_to_en_traditional_input_returns_english(
    use_case: TranslateTextUseCase,
) -> None:
    # Act
    result = use_case.execute(
        EXAMPLE_CHINESE, LanguageMap.MANDARIN, LanguageMap.ENGLISH
    )

    # Assert
    assert re.search(r"[A-Za-z]", result.target_text)
    assert not re.search(r"[一-鿿]", result.target_text)
