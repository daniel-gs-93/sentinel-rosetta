from unittest.mock import MagicMock

import pytest

from sentinel_rosetta.application.use_cases import TranslateTextUseCase
from sentinel_rosetta.domain.interfaces import TranslationInputTooLongError
from sentinel_rosetta.domain.models import LanguageMap, TranslationResult
from sentinel_rosetta.presentation import cli


@pytest.fixture
def translate_text_use_case(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    use_case = MagicMock(spec=TranslateTextUseCase)

    def build_test_use_case() -> MagicMock:
        return use_case

    monkeypatch.setattr(
        cli, "build_translate_text_use_case", build_test_use_case
    )
    return use_case


def test_main_valid_arguments_returns_zero_with_target_text_on_stdout(
    capsys: pytest.CaptureFixture[str], translate_text_use_case: MagicMock
) -> None:
    # Arrange
    translate_text_use_case.execute.return_value = TranslationResult(
        source_text="Hello",
        target_text="你好",
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
    )

    # Act
    exit_code = cli.main(["translate", "-s", "en", "-t", "zh", "Hello"])

    # Assert
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "你好"


def test_main_source_equals_target_returns_one_with_message_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Act
    exit_code = cli.main(["translate", "-s", "en", "-t", "en", "Hello"])

    # Assert
    assert exit_code == 1
    assert "source_lang and target_lang must differ" in capsys.readouterr().err


def test_main_input_too_long_returns_nonzero_exit_code(
    translate_text_use_case: MagicMock,
) -> None:
    # Arrange
    translate_text_use_case.execute.side_effect = TranslationInputTooLongError(
        "too long"
    )

    # Act
    exit_code = cli.main(["translate", "-s", "en", "-t", "zh", "Hello"])

    # Assert
    assert exit_code == 1


def test_main_startup_runtime_error_returns_one_with_message_on_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    def fail_to_build() -> TranslateTextUseCase:
        raise RuntimeError("Model binaries missing for zh")

    monkeypatch.setattr(cli, "build_translate_text_use_case", fail_to_build)

    # Act
    exit_code = cli.main(["translate", "-s", "en", "-t", "zh", "Hello"])

    # Assert
    assert exit_code == 1
    assert "Model binaries missing" in capsys.readouterr().err
