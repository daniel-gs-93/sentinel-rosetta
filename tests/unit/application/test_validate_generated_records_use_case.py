import pytest

from sentinel_rosetta.application.validate_generated_records_use_case import (
    ValidateGeneratedRecordsUseCase,
)
from sentinel_rosetta.domain.interfaces import GeneratedRecordFormatError
from tests.doubles import FakeGeneratedRecordRepository, make_record


def test_execute_returns_the_number_of_loaded_records() -> None:
    # Arrange
    records = tuple(make_record(f"id-{index}") for index in range(3))
    use_case = ValidateGeneratedRecordsUseCase(
        (FakeGeneratedRecordRepository(records),)
    )

    # Act
    record_count = use_case.execute()

    # Assert
    assert record_count == 3


def test_execute_two_repositories_returns_the_sum_of_their_records() -> None:
    # Arrange
    first_records = tuple(make_record(f"first-{index}") for index in range(3))
    second_records = tuple(make_record(f"second-{index}") for index in range(2))
    use_case = ValidateGeneratedRecordsUseCase(
        (
            FakeGeneratedRecordRepository(first_records),
            FakeGeneratedRecordRepository(second_records),
        )
    )

    # Act
    record_count = use_case.execute()

    # Assert
    assert record_count == 5


def test_execute_no_records_returns_zero() -> None:
    # Arrange
    use_case = ValidateGeneratedRecordsUseCase(
        (FakeGeneratedRecordRepository(()),)
    )

    # Act
    record_count = use_case.execute()

    # Assert
    assert record_count == 0


def test_execute_format_error_propagates() -> None:
    # Arrange
    error = GeneratedRecordFormatError("bad line")
    use_case = ValidateGeneratedRecordsUseCase(
        (FakeGeneratedRecordRepository(error=error),)
    )

    # Act / Assert
    with pytest.raises(GeneratedRecordFormatError) as error_info:
        use_case.execute()
    assert error_info.value is error
