from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel_rosetta.domain.models import LanguageMap
from sentinel_rosetta.presentation.dtos import (
    FinetuneRequest,
    TranslationRequest,
)


def test_translation_request_distinct_languages_accepts() -> None:
    # Arrange / Act
    request = TranslationRequest(
        text="Hello",
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
    )

    # Assert
    assert request.source_lang == LanguageMap.ENGLISH
    assert request.target_lang == LanguageMap.MANDARIN


@pytest.mark.parametrize(
    "language",
    [LanguageMap.ENGLISH, LanguageMap.MANDARIN],
    ids=["both-english", "both-mandarin"],
)
def test_translation_request_source_equals_target_raises_validation_error(
    language: LanguageMap,
) -> None:
    # Act / Assert
    with pytest.raises(ValidationError):
        TranslationRequest(
            text="Hello", source_lang=language, target_lang=language
        )


def test_translation_request_empty_text_raises_validation_error() -> None:
    # Act / Assert
    with pytest.raises(ValidationError):
        TranslationRequest(
            text="",
            source_lang=LanguageMap.ENGLISH,
            target_lang=LanguageMap.MANDARIN,
        )


def test_translation_request_single_character_text_accepts() -> None:
    # Arrange / Act
    request = TranslationRequest(
        text="a",
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
    )

    # Assert
    assert request.text == "a"


def test_finetune_request_valid_arguments_use_documented_defaults() -> None:
    # Act
    request = FinetuneRequest(
        source_lang=LanguageMap.ENGLISH, target_lang=LanguageMap.MANDARIN
    )

    # Assert
    assert (
        request.epochs,
        request.batch_size,
        request.learning_rate,
        request.holdout_fraction,
        request.max_length,
        request.seed,
    ) == (3, 8, 5e-5, 0.0, 128, 42)


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"learning_rate": -1e-5},
        {"target_lang": LanguageMap.ENGLISH},
    ],
    ids=[
        "epochs-zero",
        "batch-size-zero",
        "learning-rate-zero",
        "learning-rate-negative",
        "same-language",
    ],
)
def test_finetune_request_invalid_value_raises_validation_error(
    overrides: dict[str, object],
) -> None:
    # Arrange
    arguments: dict[str, object] = {
        "source_lang": LanguageMap.ENGLISH,
        "target_lang": LanguageMap.MANDARIN,
    }
    arguments.update(overrides)

    # Act / Assert
    with pytest.raises(ValidationError):
        FinetuneRequest.model_validate(arguments)


@pytest.mark.parametrize(
    ("source_lang", "target_lang", "model_name"),
    [
        (LanguageMap.ENGLISH, LanguageMap.MANDARIN, "opus-mt-en-zh"),
        (LanguageMap.MANDARIN, LanguageMap.ENGLISH, "opus-mt-zh-en"),
    ],
    ids=["en-to-zh", "zh-to-en"],
)
def test_finetune_request_resolved_dirs_follow_direction_defaults(
    source_lang: LanguageMap, target_lang: LanguageMap, model_name: str
) -> None:
    # Arrange
    request = FinetuneRequest(source_lang=source_lang, target_lang=target_lang)

    # Act / Assert
    assert request.resolved_base_model_dir() == Path("models/hf") / model_name
    assert (
        request.resolved_output_dir() == Path("models/finetuned") / model_name
    )


def test_finetune_request_explicit_dirs_override_defaults() -> None:
    # Arrange
    request = FinetuneRequest(
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
        base_model_dir=Path("my/base"),
        output_dir=Path("my/out"),
    )

    # Act / Assert
    assert request.resolved_base_model_dir() == Path("my/base")
    assert request.resolved_output_dir() == Path("my/out")


@pytest.mark.parametrize(
    "holdout_fraction",
    [-0.1, 1.0, 1.5],
    ids=["negative", "one", "above-one"],
)
def test_finetune_request_invalid_holdout_fraction_raises_validation_error(
    holdout_fraction: float,
) -> None:
    # Act / Assert
    with pytest.raises(ValidationError):
        FinetuneRequest(
            source_lang=LanguageMap.ENGLISH,
            target_lang=LanguageMap.MANDARIN,
            holdout_fraction=holdout_fraction,
        )


def test_finetune_request_holdout_fraction_just_below_one_accepts() -> None:
    # Act
    request = FinetuneRequest(
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
        holdout_fraction=0.99,
    )

    # Assert
    assert request.holdout_fraction == 0.99
