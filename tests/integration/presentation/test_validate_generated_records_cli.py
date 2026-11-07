import json
from pathlib import Path

import pytest

from sentinel_rosetta.presentation import (
    validate_generated_records_bootstrap,
    validate_generated_records_cli,
)
from tests.example_sentences import EXAMPLE_CHINESE, EXAMPLE_ENGLISH


def _line(record_id: str) -> str:
    return json.dumps(
        {
            "id": record_id,
            "zh": EXAMPLE_CHINESE,
            "en": EXAMPLE_ENGLISH,
            "origin": "llm_generated",
        },
        ensure_ascii=False,
    )


def _write_direction_file(
    generated_root: Path, direction_name: str, lines: list[str]
) -> None:
    direction_dir = generated_root / direction_name
    direction_dir.mkdir()
    (direction_dir / "llm_generated.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_validate_cli_real_run_valid_records_returns_zero_with_stdout_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    lines = [_line(f"id-{index}") for index in range(3)]
    _write_direction_file(tmp_path, "en_to_zh", lines)
    monkeypatch.setattr(
        validate_generated_records_bootstrap, "_GENERATED_ROOT", tmp_path
    )

    # Act
    exit_code = validate_generated_records_cli.main([])

    # Assert
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "checked 3 records"


def test_validate_cli_real_run_malformed_records_returns_one_with_stderr_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange: the second line has no "zh" field.
    broken = json.dumps({"id": "id-2", "en": "x", "origin": "llm_generated"})
    _write_direction_file(tmp_path, "en_to_zh", [_line("id-1"), broken])
    monkeypatch.setattr(
        validate_generated_records_bootstrap, "_GENERATED_ROOT", tmp_path
    )

    # Act
    exit_code = validate_generated_records_cli.main([])

    # Assert
    assert exit_code == 1
    assert "llm_generated.jsonl:2" in capsys.readouterr().err


def test_validate_cli_real_run_counts_records_of_every_direction_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    _write_direction_file(tmp_path, "en_to_zh", [_line("id-1"), _line("id-2")])
    _write_direction_file(tmp_path, "zh_to_en", [_line("id-3")])
    monkeypatch.setattr(
        validate_generated_records_bootstrap, "_GENERATED_ROOT", tmp_path
    )

    # Act
    exit_code = validate_generated_records_cli.main([])

    # Assert
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "checked 3 records"


def test_validate_cli_real_run_missing_generated_root_reports_zero_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    monkeypatch.setattr(
        validate_generated_records_bootstrap,
        "_GENERATED_ROOT",
        tmp_path / "missing",
    )

    # Act
    exit_code = validate_generated_records_cli.main([])

    # Assert
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "checked 0 records"
