from unittest.mock import MagicMock

import pytest

from sentinel_rosetta.application.validate_generated_records_use_case import (
    ValidateGeneratedRecordsUseCase,
)
from sentinel_rosetta.domain.interfaces import GeneratedRecordFormatError
from sentinel_rosetta.presentation import validate_generated_records_cli


@pytest.fixture
def validate_generated_records_use_case(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    use_case = MagicMock(spec=ValidateGeneratedRecordsUseCase)
    use_case.execute.return_value = 3

    def build_test_use_case() -> MagicMock:
        return use_case

    monkeypatch.setattr(
        validate_generated_records_cli,
        "build_validate_generated_records_use_case",
        build_test_use_case,
    )
    return use_case


def test_main_valid_records_returns_zero_with_summary_on_stdout(
    capsys: pytest.CaptureFixture[str],
    validate_generated_records_use_case: MagicMock,
) -> None:
    # Act
    exit_code = validate_generated_records_cli.main([])

    # Assert
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "checked 3 records"


def test_main_format_error_returns_one_with_message_on_stderr(
    capsys: pytest.CaptureFixture[str],
    validate_generated_records_use_case: MagicMock,
) -> None:
    # Arrange
    validate_generated_records_use_case.execute.side_effect = (
        GeneratedRecordFormatError(
            "data/llm_generated/en_to_zh/x.jsonl:2: invalid JSON"
        )
    )

    # Act
    exit_code = validate_generated_records_cli.main([])

    # Assert
    assert exit_code == 1
    assert (
        "data/llm_generated/en_to_zh/x.jsonl:2: invalid JSON"
        in capsys.readouterr().err
    )


def test_main_unexpected_argument_exits_with_code_two(
    validate_generated_records_use_case: MagicMock,
) -> None:
    # Act / Assert
    with pytest.raises(SystemExit) as exit_info:
        validate_generated_records_cli.main(["--records-dir", "somewhere"])
    assert exit_info.value.code == 2
