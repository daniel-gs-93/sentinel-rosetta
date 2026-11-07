import re
from pathlib import Path

from sentinel_rosetta.infrastructure.jsonl_generated_record_repository import (
    JsonlGeneratedRecordRepository,
)

_GENERATED_DIR = Path("data/llm_generated/en_to_zh")


def test_generated_directory_parses_into_records() -> None:
    # Arrange
    file_stems = {path.stem for path in _GENERATED_DIR.glob("*.jsonl")}

    # Act
    records = JsonlGeneratedRecordRepository(_GENERATED_DIR).load_all()

    # Assert
    assert len(records) >= 1
    assert {record.origin for record in records} <= file_stems


def test_generated_ids_are_unique_hex_strings() -> None:
    # Act
    records = JsonlGeneratedRecordRepository(_GENERATED_DIR).load_all()

    # Assert
    identifiers = [record.id for record in records]
    assert all(re.fullmatch(r"[0-9a-f]{32}", item) for item in identifiers)
    assert len(set(identifiers)) == len(identifiers)
