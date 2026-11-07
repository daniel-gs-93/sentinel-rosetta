from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from sentinel_rosetta.application.finetune_use_case import FineTuneModelUseCase
from sentinel_rosetta.domain.interfaces import (
    GlossaryFormatError,
    NoEvaluationDataError,
    NoTrainingDataError,
)
from sentinel_rosetta.domain.models import (
    GeneratedRecord,
    GlossaryEntry,
    LanguageMap,
    TrainingConfig,
    TrainingPair,
    TrainingResult,
)
from tests.doubles import (
    FakeGeneratedRecordRepository,
    FakeGlossaryRepository,
    FakePairTranslator,
    FakeScriptConverter,
    SpyModelTrainer,
    make_entry,
    make_record,
)

_EN = LanguageMap.ENGLISH
_ZH = LanguageMap.MANDARIN
_CONFIG = TrainingConfig(
    epochs=1, batch_size=2, learning_rate=1e-4, max_length=16, seed=42
)
_RESULT = TrainingResult(pair_count=6, train_loss=0.5, output_dir=Path("out"))
_SIMPLIFY = {"戰": "战", "車": "车"}


@dataclass
class _Harness:
    use_case: FineTuneModelUseCase
    record_repository: FakeGeneratedRecordRepository
    glossary: FakeGlossaryRepository
    trainer: SpyModelTrainer
    base_translator: FakePairTranslator
    tuned_translator: FakePairTranslator


def _build(
    records: tuple[GeneratedRecord, ...] = (),
    entries: tuple[GlossaryEntry, ...] = (),
    glossary_error: Optional[Exception] = None,
    tuned_outputs: Optional[dict[str, str]] = None,
    call_log: Optional[list[str]] = None,
) -> _Harness:
    record_repository = FakeGeneratedRecordRepository(records)
    glossary = FakeGlossaryRepository(
        entries, error=glossary_error, call_log=call_log
    )
    trainer = SpyModelTrainer(_RESULT, call_log=call_log)
    base_translator = FakePairTranslator()
    tuned_translator = FakePairTranslator(outputs_by_source=tuned_outputs)
    use_case = FineTuneModelUseCase(
        record_repository,
        glossary,
        FakeScriptConverter(_SIMPLIFY),
        trainer,
        base_translator,
        tuned_translator,
    )
    return _Harness(
        use_case,
        record_repository,
        glossary,
        trainer,
        base_translator,
        tuned_translator,
    )


def _sentence_records(count: int) -> tuple[GeneratedRecord, ...]:
    return tuple(
        make_record(
            f"id-{index}",
            zh=f"步兵戰車到達{index}。",
            en=f"The IFV arrived {index}.",
        )
        for index in range(count)
    )


def _tuned_outputs_for(count: int) -> dict[str, str]:
    return {
        f"The IFV arrived {index}.": "步兵战车到达。" for index in range(count)
    }


def test_execute_loads_every_record_once() -> None:
    # Arrange: records from three origins; none may be filtered out.
    records = (
        make_record("a", origin="llm_generated"),
        make_record("b", origin="gemini_generated"),
        make_record("c", origin="other_generated"),
    )
    harness = _build(records)

    # Act
    harness.use_case.execute(_EN, _ZH, _CONFIG)

    # Assert
    assert harness.record_repository.load_count == 1
    assert len(harness.trainer.calls[0][0]) == 3


def test_execute_delegates_training_to_the_trainer() -> None:
    # Arrange
    records = (
        make_record("a", zh="步兵戰車", en="IFV"),
        make_record("b", zh="主戰坦克", en="MBT"),
    )
    harness = _build(records)

    # Act
    harness.use_case.execute(_EN, _ZH, _CONFIG)

    # Assert
    pairs, config = harness.trainer.calls[0]
    assert pairs == (
        TrainingPair("IFV", "步兵战车"),
        TrainingPair("MBT", "主战坦克"),
    )
    assert config is _CONFIG


def test_execute_returns_trainer_result() -> None:
    # Arrange
    harness = _build(_sentence_records(2))

    # Act
    result = harness.use_case.execute(_EN, _ZH, _CONFIG)

    # Assert
    assert result is _RESULT


def test_execute_no_pairs_raises_no_training_data_error() -> None:
    # Arrange
    harness = _build(())

    # Act / Assert
    with pytest.raises(NoTrainingDataError):
        harness.use_case.execute(_EN, _ZH, _CONFIG)
    assert harness.trainer.calls == []


def test_execute_holdout_excludes_heldout_records_from_training() -> None:
    # Arrange
    harness = _build(_sentence_records(8), tuned_outputs=_tuned_outputs_for(8))

    # Act
    result = harness.use_case.execute(_EN, _ZH, _CONFIG, 0.25)

    # Assert
    trained_sources = {pair.source_text for pair in harness.trainer.calls[0][0]}
    assert result.evaluation is not None
    held_out_sources = {row.source_text for row in result.evaluation.rows}
    assert len(harness.trainer.calls[0][0]) == 6
    assert len(held_out_sources) == 2
    assert trained_sources.isdisjoint(held_out_sources)


def test_execute_holdout_attaches_report_comparing_both_checkpoints() -> None:
    # Arrange
    entry = make_entry("IFV", "infantry fighting vehicle", "步兵戰車")
    harness = _build(
        _sentence_records(8),
        entries=(entry,),
        tuned_outputs=_tuned_outputs_for(8),
    )

    # Act
    result = harness.use_case.execute(_EN, _ZH, _CONFIG, 0.25)

    # Assert
    report = result.evaluation
    assert report is not None
    assert report.sample_count == 2
    assert all(row.base_output == row.source_text for row in report.rows)
    assert all(row.tuned_output == "步兵战车到达。" for row in report.rows)
    assert (
        report.expected_term_count,
        report.base_hit_count,
        report.tuned_hit_count,
    ) == (2, 0, 2)
    assert report.tuned_score > report.base_score


def test_execute_holdout_too_small_raises_error_before_training() -> None:
    # Arrange
    harness = _build(_sentence_records(1))

    # Act / Assert
    with pytest.raises(NoEvaluationDataError):
        harness.use_case.execute(_EN, _ZH, _CONFIG, 0.25)
    assert harness.trainer.calls == []
    assert harness.base_translator.calls == []
    assert harness.tuned_translator.calls == []


def test_execute_without_holdout_skips_evaluation() -> None:
    # Arrange
    harness = _build(_sentence_records(4))

    # Act
    result = harness.use_case.execute(_EN, _ZH, _CONFIG, 0.0)

    # Assert
    assert harness.base_translator.calls == []
    assert harness.tuned_translator.calls == []
    assert harness.glossary.load_count == 0
    assert result.evaluation is None


def test_execute_holdout_loads_glossary_before_training() -> None:
    # Arrange
    call_log: list[str] = []
    harness = _build(
        _sentence_records(8),
        tuned_outputs=_tuned_outputs_for(8),
        call_log=call_log,
    )

    # Act
    harness.use_case.execute(_EN, _ZH, _CONFIG, 0.25)

    # Assert
    assert call_log.index("glossary.load") < call_log.index("trainer.train")


def test_execute_holdout_glossary_error_propagates_before_training() -> None:
    # Arrange
    error = GlossaryFormatError("bad glossary")
    harness = _build(_sentence_records(8), glossary_error=error)

    # Act / Assert
    with pytest.raises(GlossaryFormatError) as error_info:
        harness.use_case.execute(_EN, _ZH, _CONFIG, 0.25)
    assert error_info.value is error
    assert harness.trainer.calls == []


def test_execute_holdout_matches_terms_against_simplified_entries() -> None:
    # Arrange: the glossary value is Traditional, the pairs are Simplified.
    entry = make_entry("IFV", "infantry fighting vehicle", "步兵戰車")
    harness = _build(
        _sentence_records(8),
        entries=(entry,),
        tuned_outputs=_tuned_outputs_for(8),
    )

    # Act
    result = harness.use_case.execute(_EN, _ZH, _CONFIG, 0.25)

    # Assert
    assert result.evaluation is not None
    row = result.evaluation.rows[0]
    assert (row.expected_term_count, row.tuned_hit_count) == (1, 1)
