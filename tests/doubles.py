from typing import Optional

from sentinel_rosetta.domain.models import (
    GeneratedRecord,
    GlossaryEntry,
    LanguageMap,
    TrainingConfig,
    TrainingPair,
    TrainingResult,
    TranslationResult,
)


def make_record(
    record_id: str,
    zh: str = "中文",
    en: str = "English",
    origin: str = "llm_generated",
) -> GeneratedRecord:
    return GeneratedRecord(id=record_id, zh=zh, en=en, origin=origin)


def make_entry(
    abbreviation: Optional[str], full_name: str, zh: str
) -> GlossaryEntry:
    return GlossaryEntry(abbreviation=abbreviation, full_name=full_name, zh=zh)


def _replace_all(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class FakeScriptConverter:
    """A working ScriptConverter that swaps a few marker characters.

    By default it swaps the Traditional marker 繁 for the Simplified marker
    简. It records every text passed to to_simplified, and the order of its
    calls in `call_log` when one is given, so tests can check what the code
    under test handed to it and when.
    """

    def __init__(
        self,
        replacements: Optional[dict[str, str]] = None,
        call_log: Optional[list[str]] = None,
    ) -> None:
        self._to_simplified = (
            {"繁": "简"} if replacements is None else replacements
        )
        self._to_traditional = {
            simplified: traditional
            for traditional, simplified in self._to_simplified.items()
        }
        self._call_log = call_log
        self.received_texts: list[str] = []

    def to_simplified(self, text: str) -> str:
        self.received_texts.append(text)
        if self._call_log is not None:
            self._call_log.append("converter.to_simplified")
        return _replace_all(text, self._to_simplified)

    def to_traditional(self, text: str) -> str:
        if self._call_log is not None:
            self._call_log.append("converter.to_traditional")
        return _replace_all(text, self._to_traditional)


class FakeTranslationEngine:
    """A TranslationEngine that returns canned text and records its calls."""

    def __init__(
        self,
        target_text: str = "target",
        call_log: Optional[list[str]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._target_text = target_text
        self._call_log = call_log
        self._error = error
        self.calls: list[tuple[str, LanguageMap, LanguageMap]] = []

    def translate(
        self, text: str, source_lang: LanguageMap, target_lang: LanguageMap
    ) -> TranslationResult:
        self.calls.append((text, source_lang, target_lang))
        if self._call_log is not None:
            self._call_log.append("engine.translate")
        if self._error is not None:
            raise self._error
        return TranslationResult(
            source_text=text,
            target_text=self._target_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )


class FakeGeneratedRecordRepository:
    """An in-memory record repository that counts how often it was loaded."""

    def __init__(
        self,
        records: tuple[GeneratedRecord, ...] = (),
        error: Optional[Exception] = None,
    ) -> None:
        self._records = records
        self._error = error
        self.load_count = 0

    def load_all(self) -> tuple[GeneratedRecord, ...]:
        self.load_count += 1
        if self._error is not None:
            raise self._error
        return self._records


class FakeGlossaryRepository:
    """An in-memory GlossaryRepository that counts and logs its loads."""

    def __init__(
        self,
        entries: tuple[GlossaryEntry, ...] = (),
        error: Optional[Exception] = None,
        call_log: Optional[list[str]] = None,
    ) -> None:
        self._entries = entries
        self._error = error
        self._call_log = call_log
        self.load_count = 0

    def load(self) -> tuple[GlossaryEntry, ...]:
        self.load_count += 1
        if self._call_log is not None:
            self._call_log.append("glossary.load")
        if self._error is not None:
            raise self._error
        return self._entries


class SpyModelTrainer:
    """A ModelTrainer that records its calls and returns a fixed result."""

    def __init__(
        self,
        result: TrainingResult,
        call_log: Optional[list[str]] = None,
    ) -> None:
        self._result = result
        self._call_log = call_log
        self.calls: list[tuple[tuple[TrainingPair, ...], TrainingConfig]] = []

    def train(
        self, pairs: tuple[TrainingPair, ...], config: TrainingConfig
    ) -> TrainingResult:
        self.calls.append((pairs, config))
        if self._call_log is not None:
            self._call_log.append("trainer.train")
        return self._result


class FakePairTranslator:
    """A PairTranslator with canned outputs that records its calls.

    Without `outputs_by_source` it returns each source text plus `suffix`.
    """

    def __init__(
        self,
        outputs_by_source: Optional[dict[str, str]] = None,
        suffix: str = "",
    ) -> None:
        self._outputs_by_source = outputs_by_source
        self._suffix = suffix
        self.calls: list[tuple[str, ...]] = []

    def translate(self, source_texts: tuple[str, ...]) -> tuple[str, ...]:
        self.calls.append(source_texts)
        if self._outputs_by_source is None:
            return tuple(f"{text}{self._suffix}" for text in source_texts)
        return tuple(self._outputs_by_source[text] for text in source_texts)
