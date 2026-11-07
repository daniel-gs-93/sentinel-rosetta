"""Resource-aware: reads generated records from JSON Lines files."""

import json
from pathlib import Path

from sentinel_rosetta.domain.interfaces import GeneratedRecordFormatError
from sentinel_rosetta.domain.models import GeneratedRecord

_FIELDS = frozenset({"id", "zh", "en", "origin"})


class JsonlGeneratedRecordRepository:
    def __init__(self, records_dir: Path) -> None:
        self._records_dir = records_dir

    def load_all(self) -> tuple[GeneratedRecord, ...]:
        if not self._records_dir.is_dir():
            return ()

        records: list[GeneratedRecord] = []
        first_locations: dict[str, str] = {}
        for path in sorted(self._records_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    location = f"{path}:{line_number}"
                    record = _parse_record(line, path.stem, location)
                    if record.id in first_locations:
                        raise GeneratedRecordFormatError(
                            f"{location}: duplicate id {record.id!r}, "
                            f"first seen at {first_locations[record.id]}"
                        )
                    first_locations[record.id] = location
                    records.append(record)
        return tuple(records)


def _parse_record(line: str, origin: str, location: str) -> GeneratedRecord:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as e:
        raise GeneratedRecordFormatError(
            f"{location}: invalid JSON ({e.msg})"
        ) from e
    if not isinstance(payload, dict):
        raise GeneratedRecordFormatError(
            f"{location}: record must be a JSON object"
        )

    missing_fields = _FIELDS - set(payload)
    if missing_fields:
        raise GeneratedRecordFormatError(
            f"{location}: missing fields {sorted(missing_fields)}"
        )
    unexpected_fields = set(payload) - _FIELDS
    if unexpected_fields:
        raise GeneratedRecordFormatError(
            f"{location}: unexpected fields {sorted(unexpected_fields)}"
        )

    if payload["origin"] != origin:
        raise GeneratedRecordFormatError(
            f"{location}: origin {payload['origin']!r} does not match "
            f"the file name {origin!r}"
        )

    return GeneratedRecord(
        id=_require_text(payload, "id", location),
        zh=_require_text(payload, "zh", location),
        en=_require_text(payload, "en", location),
        origin=origin,
    )


def _require_text(
    payload: dict[str, object], field_name: str, location: str
) -> str:
    value = payload[field_name]
    if not isinstance(value, str) or not value.strip():
        raise GeneratedRecordFormatError(
            f"{location}: field {field_name!r} must be a non-empty string"
        )
    return value
