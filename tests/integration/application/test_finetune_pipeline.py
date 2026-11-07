import json
from pathlib import Path
from typing import Optional

from sentinel_rosetta.application.finetune_use_case import FineTuneModelUseCase
from sentinel_rosetta.domain.models import (
    LanguageMap,
    TrainingConfig,
    TrainingPair,
    TrainingResult,
)
from sentinel_rosetta.infrastructure.json_glossary_repository import (
    JsonGlossaryRepository,
)
from sentinel_rosetta.infrastructure.jsonl_generated_record_repository import (
    JsonlGeneratedRecordRepository,
)
from sentinel_rosetta.infrastructure.opencc_adapter import OpenCcAdapter
from tests.doubles import FakePairTranslator, SpyModelTrainer
from tests.example_sentences import (
    EXAMPLE_CHINESE,
    EXAMPLE_CHINESE_SIMPLIFIED,
    EXAMPLE_ENGLISH,
)

_CONFIG = TrainingConfig(
    epochs=1, batch_size=2, learning_rate=1e-4, max_length=64, seed=1
)
_RESULT = TrainingResult(pair_count=0, train_loss=0.0, output_dir=Path("out"))


def _write_generated_records(
    records_dir: Path, file_stem: str, records: list[tuple[str, str, str]]
) -> None:
    lines = [
        json.dumps(
            {"id": record_id, "zh": zh, "en": en, "origin": file_stem},
            ensure_ascii=False,
        )
        for record_id, zh, en in records
    ]
    (records_dir / f"{file_stem}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _numbered_records(count: int, prefix: str) -> list[tuple[str, str, str]]:
    return [
        (
            f"{prefix}{index}",
            f"步兵戰車第{index}號已經到達基地。",
            f"The IFV number {index} has arrived at base.",
        )
        for index in range(count)
    ]


def _use_case(
    records_dir: Path,
    trainer: SpyModelTrainer,
    glossary_path: Optional[Path] = None,
    base_translator: Optional[FakePairTranslator] = None,
    tuned_translator: Optional[FakePairTranslator] = None,
) -> FineTuneModelUseCase:
    return FineTuneModelUseCase(
        JsonlGeneratedRecordRepository(records_dir),
        JsonGlossaryRepository(glossary_path or records_dir / "no-glossary"),
        OpenCcAdapter(),
        trainer,
        base_translator or FakePairTranslator(),
        tuned_translator or FakePairTranslator(),
    )


def test_pipeline_traditional_records_reach_trainer_as_simplified_pairs(
    tmp_path: Path,
) -> None:
    # Arrange
    _write_generated_records(
        tmp_path,
        "llm_generated",
        [("a1", EXAMPLE_CHINESE, EXAMPLE_ENGLISH)],
    )
    trainer = SpyModelTrainer(_RESULT)

    # Act
    _use_case(tmp_path, trainer).execute(
        LanguageMap.ENGLISH, LanguageMap.MANDARIN, _CONFIG
    )

    # Assert
    (pairs, _config) = trainer.calls[0]
    assert pairs == (TrainingPair(EXAMPLE_ENGLISH, EXAMPLE_CHINESE_SIMPLIFIED),)


def test_pipeline_every_jsonl_file_contributes_pairs_to_the_trainer(
    tmp_path: Path,
) -> None:
    # Arrange
    _write_generated_records(
        tmp_path, "llm_generated", _numbered_records(3, "c")
    )
    _write_generated_records(
        tmp_path, "gemini_generated", _numbered_records(2, "g")
    )
    trainer = SpyModelTrainer(_RESULT)

    # Act
    _use_case(tmp_path, trainer).execute(
        LanguageMap.ENGLISH, LanguageMap.MANDARIN, _CONFIG
    )

    # Assert
    assert len(trainer.calls[0][0]) == 5


def test_pipeline_holdout_sentences_never_reach_the_trainer(
    tmp_path: Path,
) -> None:
    # Arrange: holdout is read from the glossary only for terms, so an
    # empty glossary file keeps this test about the split.
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text('{"entries": []}', encoding="utf-8")
    _write_generated_records(
        tmp_path, "llm_generated", _numbered_records(8, "c")
    )
    trainer = SpyModelTrainer(_RESULT)
    use_case = _use_case(
        tmp_path,
        trainer,
        glossary_path,
        FakePairTranslator(suffix=" base"),
        FakePairTranslator(suffix=" tuned"),
    )

    # Act
    result = use_case.execute(
        LanguageMap.ENGLISH, LanguageMap.MANDARIN, _CONFIG, 0.25
    )

    # Assert
    (trained_pairs, _config) = trainer.calls[0]
    trained_sources = {pair.source_text for pair in trained_pairs}
    assert result.evaluation is not None
    heldout_sources = {row.source_text for row in result.evaluation.rows}
    assert (len(trained_sources), len(heldout_sources)) == (6, 2)
    assert trained_sources & heldout_sources == set()


def test_pipeline_holdout_report_counts_terms_from_the_glossary_file(
    tmp_path: Path,
) -> None:
    # Arrange: four records and a 0.25 hold-out leave exactly one held-out
    # sentence, whose reference holds the Simplified glossary value.
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "abbreviation": "IFV",
                        "full_name": "infantry fighting vehicle",
                        "zh": "步兵戰車",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    records = _numbered_records(4, "c")
    _write_generated_records(tmp_path, "llm_generated", records)
    tuned_outputs = {en: "步兵战车到达。" for _id, _zh, en in records}
    trainer = SpyModelTrainer(_RESULT)
    use_case = _use_case(
        tmp_path,
        trainer,
        glossary_path,
        tuned_translator=FakePairTranslator(outputs_by_source=tuned_outputs),
    )

    # Act
    result = use_case.execute(
        LanguageMap.ENGLISH, LanguageMap.MANDARIN, _CONFIG, 0.25
    )

    # Assert
    assert result.evaluation is not None
    report = result.evaluation
    assert (
        report.expected_term_count,
        report.base_hit_count,
        report.tuned_hit_count,
    ) == (1, 0, 1)
