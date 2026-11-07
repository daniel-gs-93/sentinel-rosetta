"""Resource-aware: builds the translation use case from its adapters."""

from pathlib import Path

from sentinel_rosetta.application.use_cases import TranslateTextUseCase
from sentinel_rosetta.infrastructure.ctranslate2_adapter import (
    CTranslate2Adapter,
)
from sentinel_rosetta.infrastructure.opencc_adapter import OpenCcAdapter

_MODELS_DIR = Path("models")


def build_translate_text_use_case() -> TranslateTextUseCase:
    translation_engine = CTranslate2Adapter(models_dir=_MODELS_DIR)
    script_converter = OpenCcAdapter()
    return TranslateTextUseCase(translation_engine, script_converter)
