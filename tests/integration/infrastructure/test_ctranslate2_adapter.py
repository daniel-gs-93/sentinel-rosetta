import shutil
from pathlib import Path

import pytest

from sentinel_rosetta.domain.interfaces import TranslationInputTooLongError
from sentinel_rosetta.domain.models import LanguageMap
from sentinel_rosetta.infrastructure.ctranslate2_adapter import (
    CTranslate2Adapter,
)
from tests.model_environment import missing_translation_models

_MODELS_DIR = Path("models")
_MISSING = missing_translation_models()

pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"translation models missing: {_MISSING}"
)


@pytest.fixture(scope="module")
def ctranslate2_adapter() -> CTranslate2Adapter:
    # Loading real CTranslate2 models is costly, and translate() never
    # mutates adapter state, so this module's tests share one instance
    # instead of reloading the weights for each test.
    return CTranslate2Adapter(models_dir=_MODELS_DIR)


def test_translate_short_text_returns_nonempty_target_text(
    ctranslate2_adapter: CTranslate2Adapter,
) -> None:
    # Act
    result = ctranslate2_adapter.translate(
        "Hello", LanguageMap.ENGLISH, LanguageMap.MANDARIN
    )

    # Assert
    assert result.target_text != ""


def test_translate_text_at_512_tokens_succeeds(
    ctranslate2_adapter: CTranslate2Adapter,
) -> None:
    # Arrange: the tokenizer appends one EOS token, so 511 repeats of a
    # single-token word lands exactly on the 512-token boundary.
    text = "word " * 511

    # Act / Assert
    ctranslate2_adapter.translate(
        text, LanguageMap.ENGLISH, LanguageMap.MANDARIN
    )


def test_translate_text_over_512_tokens_raises_translation_input_too_long_error(
    ctranslate2_adapter: CTranslate2Adapter,
) -> None:
    # Arrange: one repeat more than the at-boundary case lands exactly one
    # token past the 512-token boundary (513 tokens).
    text = "word " * 512

    # Act / Assert
    with pytest.raises(TranslationInputTooLongError):
        ctranslate2_adapter.translate(
            text, LanguageMap.ENGLISH, LanguageMap.MANDARIN
        )


def test_adapter_missing_one_direction_raises_runtime_error(
    tmp_path: Path,
) -> None:
    # Arrange: only the en-zh directories exist; zh-en is missing.
    for name in ("opus-mt-en-zh", "opus-mt-en-zh-ct2"):
        shutil.copytree(_MODELS_DIR / name, tmp_path / name)

    # Act / Assert
    with pytest.raises(RuntimeError, match="Model binaries missing"):
        CTranslate2Adapter(models_dir=tmp_path)
