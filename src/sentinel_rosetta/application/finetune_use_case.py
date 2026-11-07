"""Pure logic: orchestrates fine-tuning through injected interfaces."""

import dataclasses

from sentinel_rosetta.domain.evaluation import build_evaluation_report
from sentinel_rosetta.domain.glossary_terms import simplify_entries
from sentinel_rosetta.domain.interfaces import (
    GeneratedRecordRepository,
    GlossaryRepository,
    ModelTrainer,
    NoEvaluationDataError,
    NoTrainingDataError,
    PairTranslator,
    ScriptConverter,
)
from sentinel_rosetta.domain.models import (
    GlossaryEntry,
    LanguageMap,
    TrainingConfig,
    TrainingPair,
    TrainingResult,
)
from sentinel_rosetta.domain.training_data import (
    build_training_pairs,
    split_holdout,
)


class FineTuneModelUseCase:
    def __init__(
        self,
        record_repository: GeneratedRecordRepository,
        glossary_repository: GlossaryRepository,
        script_converter: ScriptConverter,
        model_trainer: ModelTrainer,
        base_translator: PairTranslator,
        tuned_translator: PairTranslator,
    ) -> None:
        self._record_repository = record_repository
        self._glossary_repository = glossary_repository
        self._script_converter = script_converter
        self._model_trainer = model_trainer
        self._base_translator = base_translator
        self._tuned_translator = tuned_translator

    def execute(
        self,
        source_lang: LanguageMap,
        target_lang: LanguageMap,
        config: TrainingConfig,
        holdout_fraction: float = 0.0,
    ) -> TrainingResult:
        records = self._record_repository.load_all()
        train_records, holdout_records = split_holdout(
            records, holdout_fraction, config.seed
        )
        training_pairs = build_training_pairs(
            train_records, source_lang, target_lang, self._script_converter
        )

        holdout_pairs: tuple[TrainingPair, ...] = ()
        glossary_entries: tuple[GlossaryEntry, ...] = ()
        if holdout_fraction > 0:
            holdout_pairs = build_training_pairs(
                holdout_records,
                source_lang,
                target_lang,
                self._script_converter,
            )
            glossary_entries = simplify_entries(
                self._glossary_repository.load(), self._script_converter
            )

        if not training_pairs:
            raise NoTrainingDataError(
                f"No training pairs remain for {source_lang.value}_to_"
                f"{target_lang.value}; add records to that direction "
                "folder of data/llm_generated/."
            )
        if holdout_fraction > 0 and not holdout_pairs:
            raise NoEvaluationDataError(
                "The hold-out set is empty; add records or raise "
                "--holdout-fraction."
            )

        result = self._model_trainer.train(training_pairs, config)
        if not holdout_pairs:
            return result

        source_texts = tuple(pair.source_text for pair in holdout_pairs)
        base_outputs = self._base_translator.translate(source_texts)
        tuned_outputs = self._tuned_translator.translate(source_texts)
        report = build_evaluation_report(
            holdout_pairs,
            base_outputs,
            tuned_outputs,
            glossary_entries,
            source_lang,
            target_lang,
        )
        return dataclasses.replace(result, evaluation=report)
