import json
from pathlib import Path

import pytest

from sentinel_rosetta.domain.interfaces import GeneratedRecordFormatError
from sentinel_rosetta.domain.models import GeneratedRecord
from sentinel_rosetta.infrastructure.jsonl_generated_record_repository import (
    JsonlGeneratedRecordRepository,
)


def _line(
    record_id: str,
    zh: str = "中文",
    en: str = "English",
    origin: str = "llm_generated",
) -> str:
    return json.dumps(
        {"id": record_id, "zh": zh, "en": en, "origin": origin},
        ensure_ascii=False,
    )


def _write(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_all_single_file_returns_records_in_file_order(
    tmp_path: Path,
) -> None:
    # Arrange
    _write(
        tmp_path / "llm_generated.jsonl",
        _line("a", zh="步兵戰車", en="IFV"),
        _line("b", zh="主戰坦克", en="MBT"),
    )

    # Act
    records = JsonlGeneratedRecordRepository(tmp_path).load_all()

    # Assert
    assert records == (
        GeneratedRecord("a", "步兵戰車", "IFV", "llm_generated"),
        GeneratedRecord("b", "主戰坦克", "MBT", "llm_generated"),
    )


def test_load_all_two_files_returns_records_ordered_by_file_name_then_line(
    tmp_path: Path,
) -> None:
    # Arrange: written in reverse name order to show the sort is by name.
    _write(
        tmp_path / "b_generated.jsonl",
        _line("b1", origin="b_generated"),
        _line("b2", origin="b_generated"),
    )
    _write(
        tmp_path / "a_generated.jsonl",
        _line("a1", origin="a_generated"),
        _line("a2", origin="a_generated"),
    )

    # Act
    records = JsonlGeneratedRecordRepository(tmp_path).load_all()

    # Assert
    assert [(record.id, record.origin) for record in records] == [
        ("a1", "a_generated"),
        ("a2", "a_generated"),
        ("b1", "b_generated"),
        ("b2", "b_generated"),
    ]


def test_load_all_ignores_files_without_jsonl_extension(
    tmp_path: Path,
) -> None:
    # Arrange
    _write(tmp_path / "llm_generated.jsonl", _line("a"))
    (tmp_path / "README.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not json\n", encoding="utf-8")

    # Act
    records = JsonlGeneratedRecordRepository(tmp_path).load_all()

    # Assert
    assert [record.id for record in records] == ["a"]


@pytest.mark.parametrize(
    "directory_name",
    ["empty", "missing"],
    ids=["empty-dir", "missing-dir"],
)
def test_load_all_directory_without_jsonl_files_returns_empty_tuple(
    tmp_path: Path, directory_name: str
) -> None:
    # Arrange
    records_dir = tmp_path / directory_name
    if directory_name == "empty":
        records_dir.mkdir()

    # Act
    records = JsonlGeneratedRecordRepository(records_dir).load_all()

    # Assert
    assert records == ()


def test_load_all_blank_lines_are_skipped(tmp_path: Path) -> None:
    # Arrange
    _write(
        tmp_path / "llm_generated.jsonl",
        "",
        _line("a"),
        "   ",
        _line("b"),
        "",
    )

    # Act
    records = JsonlGeneratedRecordRepository(tmp_path).load_all()

    # Assert
    assert [record.id for record in records] == ["a", "b"]


def test_load_all_invalid_json_error_locates_the_failing_line(
    tmp_path: Path,
) -> None:
    # Arrange
    path = tmp_path / "llm_generated.jsonl"
    _write(path, _line("a"), "{not json")

    # Act / Assert
    with pytest.raises(GeneratedRecordFormatError) as error_info:
        JsonlGeneratedRecordRepository(tmp_path).load_all()
    assert f"{path}:2" in str(error_info.value)


@pytest.mark.parametrize(
    "raw_line",
    [
        '["not", "an", "object"]',
        json.dumps({"id": "a", "zh": "中", "origin": "llm_generated"}),
        _line("a", zh="   "),
        _line("a", en="   "),
        json.dumps(
            {
                "id": "a",
                "zh": "中",
                "en": "E",
                "origin": "llm_generated",
                "note": "extra",
            }
        ),
        json.dumps(
            {
                "id": "a",
                "zh": "中",
                "en": "E",
                "origin": "llm_generated",
                "model": "opus-mt-en-zh",
            }
        ),
        json.dumps(
            {"id": 123, "zh": "中", "en": "E", "origin": "llm_generated"}
        ),
        _line("a", origin="other_generated"),
    ],
    ids=[
        "not-an-object",
        "missing-field",
        "blank-zh",
        "blank-en",
        "unknown-field",
        "model-field",
        "non-string-id",
        "origin-mismatch",
    ],
)
def test_load_all_invalid_record_raises_generated_record_format_error(
    tmp_path: Path, raw_line: str
) -> None:
    # Arrange
    _write(tmp_path / "llm_generated.jsonl", raw_line)

    # Act / Assert
    with pytest.raises(GeneratedRecordFormatError):
        JsonlGeneratedRecordRepository(tmp_path).load_all()


def test_load_all_duplicate_id_in_one_file_raises_generated_record_format_error(
    tmp_path: Path,
) -> None:
    # Arrange
    _write(tmp_path / "llm_generated.jsonl", _line("a"), _line("a"))

    # Act / Assert
    with pytest.raises(GeneratedRecordFormatError):
        JsonlGeneratedRecordRepository(tmp_path).load_all()


def test_load_all_duplicate_id_across_files_raises_error_naming_both_files(
    tmp_path: Path,
) -> None:
    # Arrange
    _write(
        tmp_path / "a_generated.jsonl", _line("shared", origin="a_generated")
    )
    _write(
        tmp_path / "b_generated.jsonl", _line("shared", origin="b_generated")
    )

    # Act / Assert
    with pytest.raises(GeneratedRecordFormatError) as error_info:
        JsonlGeneratedRecordRepository(tmp_path).load_all()
    message = str(error_info.value)
    assert "a_generated.jsonl" in message
    assert "b_generated.jsonl" in message


def test_load_all_same_en_in_two_files_is_not_an_error(tmp_path: Path) -> None:
    # Arrange
    _write(
        tmp_path / "a_generated.jsonl",
        _line("a1", en="Same sentence.", origin="a_generated"),
    )
    _write(
        tmp_path / "b_generated.jsonl",
        _line("b1", en="Same sentence.", origin="b_generated"),
    )

    # Act
    records = JsonlGeneratedRecordRepository(tmp_path).load_all()

    # Assert
    assert [record.id for record in records] == ["a1", "b1"]
