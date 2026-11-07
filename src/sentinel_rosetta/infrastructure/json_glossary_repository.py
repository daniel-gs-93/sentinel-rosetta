"""Resource-aware: reads reference glossary entries from a JSON file."""

import json
from pathlib import Path

from sentinel_rosetta.domain.interfaces import GlossaryFormatError
from sentinel_rosetta.domain.models import GlossaryEntry


class JsonGlossaryRepository:
    def __init__(self, glossary_path: Path) -> None:
        self._glossary_path = glossary_path

    def load(self) -> tuple[GlossaryEntry, ...]:
        path = self._glossary_path
        try:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError as e:
            raise GlossaryFormatError(f"{path}: file not found") from e
        except json.JSONDecodeError as e:
            raise GlossaryFormatError(f"{path}: invalid JSON ({e.msg})") from e

        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise GlossaryFormatError(f"{path}: missing 'entries' list")
        return tuple(
            _parse_entry(item, f"{path}: entry {index}")
            for index, item in enumerate(entries, start=1)
        )


def _parse_entry(item: object, location: str) -> GlossaryEntry:
    if not isinstance(item, dict):
        raise GlossaryFormatError(f"{location}: entry must be a JSON object")
    missing_fields = {"abbreviation", "full_name", "zh"} - set(item)
    if missing_fields:
        raise GlossaryFormatError(
            f"{location}: missing fields {sorted(missing_fields)}"
        )

    abbreviation = item["abbreviation"]
    if abbreviation is not None and not _is_text(abbreviation):
        raise GlossaryFormatError(
            f"{location}: 'abbreviation' must be null or a non-empty string"
        )
    return GlossaryEntry(
        abbreviation=abbreviation,
        full_name=_require_text(item, "full_name", location),
        zh=_require_text(item, "zh", location),
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_text(
    item: dict[str, object], field_name: str, location: str
) -> str:
    value = item[field_name]
    if not isinstance(value, str) or not value.strip():
        raise GlossaryFormatError(
            f"{location}: field {field_name!r} must be a non-empty string"
        )
    return value
