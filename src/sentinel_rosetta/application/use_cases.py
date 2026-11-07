"""Pure logic: orchestrates translation through injected interfaces."""

from sentinel_rosetta.domain.interfaces import (
    ScriptConverter,
    TranslationEngine,
)
from sentinel_rosetta.domain.models import LanguageMap, TranslationResult


class TranslateTextUseCase:
    def __init__(
        self,
        translation_engine: TranslationEngine,
        script_converter: ScriptConverter,
    ) -> None:
        self._translation_engine = translation_engine
        self._script_converter = script_converter

    def execute(
        self, text: str, source_lang: LanguageMap, target_lang: LanguageMap
    ) -> TranslationResult:
        if source_lang == LanguageMap.ENGLISH:
            target_text = self._translate_en_to_zh(
                text, source_lang, target_lang
            )
        else:
            target_text = self._translate_zh_to_en(
                text, source_lang, target_lang
            )

        return TranslationResult(
            source_text=text,
            target_text=target_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _translate_en_to_zh(
        self, text: str, source_lang: LanguageMap, target_lang: LanguageMap
    ) -> str:
        result = self._translation_engine.translate(
            text, source_lang, target_lang
        )
        return self._script_converter.to_traditional(result.target_text)

    def _translate_zh_to_en(
        self, text: str, source_lang: LanguageMap, target_lang: LanguageMap
    ) -> str:
        simplified = self._script_converter.to_simplified(text)
        result = self._translation_engine.translate(
            simplified, source_lang, target_lang
        )
        return result.target_text
