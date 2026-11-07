"""Pure logic: ports and error types owned by the domain."""

from typing import Protocol

from sentinel_rosetta.domain.models import (
    GeneratedRecord,
    GlossaryEntry,
    LanguageMap,
    TrainingConfig,
    TrainingPair,
    TrainingResult,
    TranslationResult,
)


class TranslationInputTooLongError(ValueError):
    pass


class GeneratedRecordFormatError(ValueError):
    pass


class GlossaryFormatError(ValueError):
    pass


class NoTrainingDataError(ValueError):
    pass


class NoEvaluationDataError(ValueError):
    pass


class TrainingEnvironmentError(RuntimeError):
    pass


class TranslationEngine(Protocol):
    def translate(
        self, text: str, source_lang: LanguageMap, target_lang: LanguageMap
    ) -> TranslationResult: ...


class ScriptConverter(Protocol):
    def to_traditional(self, text: str) -> str: ...

    def to_simplified(self, text: str) -> str: ...


class GeneratedRecordRepository(Protocol):
    def load_all(self) -> tuple[GeneratedRecord, ...]: ...


class GlossaryRepository(Protocol):
    def load(self) -> tuple[GlossaryEntry, ...]: ...


class ModelTrainer(Protocol):
    def train(
        self, pairs: tuple[TrainingPair, ...], config: TrainingConfig
    ) -> TrainingResult: ...


class PairTranslator(Protocol):
    def translate(self, source_texts: tuple[str, ...]) -> tuple[str, ...]: ...
