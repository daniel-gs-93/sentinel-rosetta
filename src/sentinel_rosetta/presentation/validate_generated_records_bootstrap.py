"""Resource-aware: wires the validation use case to the record files."""

from pathlib import Path

from sentinel_rosetta.application.validate_generated_records_use_case import (
    ValidateGeneratedRecordsUseCase,
)
from sentinel_rosetta.infrastructure.jsonl_generated_record_repository import (
    JsonlGeneratedRecordRepository,
)

_GENERATED_ROOT = Path("data/llm_generated")


def build_validate_generated_records_use_case() -> (
    ValidateGeneratedRecordsUseCase
):
    return ValidateGeneratedRecordsUseCase(
        tuple(
            JsonlGeneratedRecordRepository(direction_dir)
            for direction_dir in _direction_dirs()
        )
    )


def _direction_dirs() -> list[Path]:
    if not _GENERATED_ROOT.is_dir():
        return []
    return sorted(
        direction_dir
        for direction_dir in _GENERATED_ROOT.iterdir()
        if direction_dir.is_dir()
    )
