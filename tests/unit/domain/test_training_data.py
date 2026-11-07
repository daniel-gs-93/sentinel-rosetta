import functools
import hashlib

import pytest

from sentinel_rosetta.domain.models import (
    GeneratedRecord,
    LanguageMap,
    TrainingPair,
)
from sentinel_rosetta.domain.training_data import (
    build_training_pairs,
    split_holdout,
)
from tests.doubles import FakeScriptConverter, make_record


def test_build_pairs_en_to_zh_source_is_english_target_is_simplified() -> None:
    # Arrange
    records = (make_record("r1", zh="繁中", en="Hello"),)

    # Act
    pairs = build_training_pairs(
        records,
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
        FakeScriptConverter(),
    )

    # Assert
    assert pairs == (TrainingPair("Hello", "简中"),)


def test_build_pairs_zh_to_en_source_is_simplified_target_is_english() -> None:
    # Arrange
    records = (make_record("r1", zh="繁中", en="Hello"),)

    # Act
    pairs = build_training_pairs(
        records,
        LanguageMap.MANDARIN,
        LanguageMap.ENGLISH,
        FakeScriptConverter(),
    )

    # Assert
    assert pairs == (TrainingPair("简中", "Hello"),)


@pytest.mark.parametrize(
    ("raw_en", "raw_zh", "expected_en", "expected_zh"),
    [
        ("  Hello world  ", " 中文 ", "Hello world", "中文"),
        ("Hello   world", "中   文", "Hello world", "中 文"),
        ("Hello\nworld", "中\n文", "Hello world", "中 文"),
        ("Hello\tworld", "中\t文", "Hello world", "中 文"),
    ],
    ids=["leading-trailing", "repeated-spaces", "newline", "tab"],
)
def test_build_pairs_whitespace_runs_collapse_to_single_space(
    raw_en: str, raw_zh: str, expected_en: str, expected_zh: str
) -> None:
    # Arrange
    records = (make_record("r1", zh=raw_zh, en=raw_en),)

    # Act
    pairs = build_training_pairs(
        records,
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
        FakeScriptConverter(),
    )

    # Assert
    assert pairs == (TrainingPair(expected_en, expected_zh),)


@pytest.mark.parametrize(
    ("raw_en", "raw_zh"),
    [
        ("", "中文"),
        ("Hello", ""),
        ("   ", "中文"),
        ("Hello", " \n "),
    ],
    ids=["empty-en", "empty-zh", "whitespace-en", "whitespace-zh"],
)
def test_build_pairs_blank_side_drops_the_record(
    raw_en: str, raw_zh: str
) -> None:
    # Arrange
    records = (make_record("r1", zh=raw_zh, en=raw_en),)

    # Act
    pairs = build_training_pairs(
        records,
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
        FakeScriptConverter(),
    )

    # Assert
    assert pairs == ()


def test_build_pairs_keeps_duplicate_records_in_input_order() -> None:
    # Arrange
    records = (
        make_record("a", zh="甲", en="A"),
        make_record("b", zh="乙", en="B"),
        make_record("c", zh="甲", en="A"),
    )

    # Act
    pairs = build_training_pairs(
        records,
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
        FakeScriptConverter(),
    )

    # Assert
    assert pairs == (
        TrainingPair("A", "甲"),
        TrainingPair("B", "乙"),
        TrainingPair("A", "甲"),
    )


def test_build_pairs_empty_records_returns_empty_tuple() -> None:
    # Act
    pairs = build_training_pairs(
        (),
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
        FakeScriptConverter(),
    )

    # Assert
    assert pairs == ()


@pytest.mark.parametrize(
    "language",
    [LanguageMap.ENGLISH, LanguageMap.MANDARIN],
    ids=["both-english", "both-mandarin"],
)
def test_build_pairs_identical_languages_raises_value_error(
    language: LanguageMap,
) -> None:
    # Act / Assert
    with pytest.raises(ValueError):
        build_training_pairs(
            (make_record("r1"),), language, language, FakeScriptConverter()
        )


def test_build_pairs_converter_receives_normalized_text() -> None:
    # Arrange: the second record is dropped for its blank English side, so
    # the converter must be called once, for the first record only.
    records = (
        make_record("r1", zh="  繁  中 ", en="  Hello  "),
        make_record("r2", zh="繁", en="   "),
    )
    converter = FakeScriptConverter()

    # Act
    build_training_pairs(
        records, LanguageMap.ENGLISH, LanguageMap.MANDARIN, converter
    )

    # Assert
    assert converter.received_texts == ["繁 中"]


def _records(count: int) -> tuple[GeneratedRecord, ...]:
    return tuple(make_record(f"id-{index}") for index in range(count))


def _record_ids(records: tuple[GeneratedRecord, ...]) -> set[str]:
    return {record.id for record in records}


def test_split_zero_fraction_keeps_every_record_for_training() -> None:
    # Arrange
    records = _records(5)

    # Act
    train_records, holdout_records = split_holdout(records, 0.0, 42)

    # Assert
    assert (train_records, holdout_records) == (records, ())


@pytest.mark.parametrize(
    ("record_count", "holdout_fraction", "expected_size"),
    [(40, 0.25, 10), (10, 0.25, 3), (3, 0.5, 2), (1, 0.25, 0), (2, 0.99, 2)],
    ids=[
        "exact",
        "half-rounds-up",
        "one-and-a-half",
        "below-half",
        "almost-all",
    ],
)
def test_split_holdout_size_rounds_half_up(
    record_count: int, holdout_fraction: float, expected_size: int
) -> None:
    # Arrange
    records = _records(record_count)

    # Act
    _train_records, holdout_records = split_holdout(
        records, holdout_fraction, 42
    )

    # Assert
    assert len(holdout_records) == expected_size


def test_split_result_is_independent_of_record_order() -> None:
    # Arrange
    records = _records(20)

    # Act
    _train_a, holdout_a = split_holdout(records, 0.25, 42)
    _train_b, holdout_b = split_holdout(tuple(reversed(records)), 0.25, 42)

    # Assert
    assert _record_ids(holdout_a) == _record_ids(holdout_b)


def test_split_partitions_records_without_loss_or_overlap() -> None:
    # Arrange
    records = _records(12)

    # Act
    train_records, holdout_records = split_holdout(records, 0.25, 42)

    # Assert
    assert _record_ids(train_records) | _record_ids(holdout_records) == (
        _record_ids(records)
    )
    assert _record_ids(train_records) & _record_ids(holdout_records) == set()
    assert train_records == tuple(
        record for record in records if record in train_records
    )
    assert holdout_records == tuple(
        record for record in records if record in holdout_records
    )


def test_split_different_seeds_select_different_holdouts() -> None:
    # Arrange
    records = _records(40)

    # Act
    _train_a, holdout_a = split_holdout(records, 0.25, 1)
    _train_b, holdout_b = split_holdout(records, 0.25, 2)

    # Assert
    assert _record_ids(holdout_a) != _record_ids(holdout_b)


def _hash_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def test_split_holdout_selects_by_id_hash_regardless_of_origin() -> None:
    # Arrange: records from two origins; the expected hold-out is the first
    # ten ids in sha256(f"{seed}:{id}") order, whatever their origin.
    seed = 7
    records = tuple(
        make_record(f"id-{index}", origin=origin)
        for index, origin in enumerate(
            ["llm_generated", "gemini_generated"] * 20
        )
    )
    ranked_ids = sorted(
        (record.id for record in records),
        key=functools.partial(_hash_rank, seed),
    )

    # Act
    _train_records, holdout_records = split_holdout(records, 0.25, seed)

    # Assert
    assert _record_ids(holdout_records) == set(ranked_ids[:10])
