import json
from pathlib import Path
from typing import Optional

import pytest

from sentinel_rosetta.domain.interfaces import GlossaryFormatError
from sentinel_rosetta.domain.models import GlossaryEntry
from sentinel_rosetta.infrastructure.json_glossary_repository import (
    JsonGlossaryRepository,
)


def _entry(
    abbreviation: Optional[str] = "IFV",
    full_name: str = "infantry fighting vehicle",
    zh: str = "步兵戰車",
) -> dict[str, object]:
    return {
        "abbreviation": abbreviation,
        "full_name": full_name,
        "zh": zh,
        "source": "atp-7-100.3-glossary",
    }


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_valid_glossary_returns_entries_in_file_order(
    tmp_path: Path,
) -> None:
    # Arrange
    second = _entry("PLAAF", "People's Liberation Army Air Force", "解放軍空軍")
    second["note"] = "全稱：中國人民解放軍空軍"
    path = _write(
        tmp_path / "glossary.json",
        {"meta": {"description": "x"}, "entries": [_entry(), second]},
    )

    # Act
    entries = JsonGlossaryRepository(path).load()

    # Assert
    assert entries == (
        GlossaryEntry("IFV", "infantry fighting vehicle", "步兵戰車"),
        GlossaryEntry(
            "PLAAF", "People's Liberation Army Air Force", "解放軍空軍"
        ),
    )


def test_load_glossary_null_abbreviation_becomes_none(tmp_path: Path) -> None:
    # Arrange
    path = _write(
        tmp_path / "glossary.json",
        {"entries": [_entry(None, "course of action", "行動方案")]},
    )

    # Act
    entries = JsonGlossaryRepository(path).load()

    # Assert
    assert entries[0].abbreviation is None


def _write_raw(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "case",
    [
        "missing-file",
        "not-json",
        "missing-entries",
        "entry-not-object",
        "blank-full-name",
        "blank-zh",
        "abbreviation-wrong-type",
    ],
)
def test_load_invalid_glossary_raises_glossary_format_error(
    tmp_path: Path, case: str
) -> None:
    # Arrange
    path = tmp_path / "glossary.json"
    if case == "not-json":
        _write_raw(path, "{not json")
    elif case == "missing-entries":
        _write(path, {"meta": {}})
    elif case == "entry-not-object":
        _write(path, {"entries": ["IFV"]})
    elif case == "blank-full-name":
        _write(path, {"entries": [_entry(full_name="  ")]})
    elif case == "blank-zh":
        _write(path, {"entries": [_entry(zh="")]})
    elif case == "abbreviation-wrong-type":
        _write(path, {"entries": [_entry(abbreviation=5)]})  # type: ignore[arg-type]

    # Act / Assert
    with pytest.raises(GlossaryFormatError):
        JsonGlossaryRepository(path).load()


def test_load_invalid_entry_error_locates_the_entry(tmp_path: Path) -> None:
    # Arrange
    broken = _entry()
    del broken["zh"]
    path = _write(tmp_path / "glossary.json", {"entries": [_entry(), broken]})

    # Act / Assert
    with pytest.raises(GlossaryFormatError) as error_info:
        JsonGlossaryRepository(path).load()
    message = str(error_info.value)
    assert str(path) in message
    assert "entry 2" in message
