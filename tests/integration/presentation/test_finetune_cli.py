import json
import re
from pathlib import Path

import pytest

from sentinel_rosetta.presentation import finetune_bootstrap, finetune_cli
from tests.training_environment import missing_training_prerequisites

_MISSING = missing_training_prerequisites()

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"fine-tuning prerequisites missing: {_MISSING}",
)

_RECORD_COUNT = 8


def _write_generated_records(generated_root: Path) -> None:
    direction_dir = generated_root / "en_to_zh"
    direction_dir.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "id": f"id-{index}",
                "zh": f"步兵戰車第{index}號已經到達基地。",
                "en": f"The IFV number {index} has arrived at base.",
                "origin": "llm_generated",
            },
            ensure_ascii=False,
        )
        for index in range(_RECORD_COUNT)
    ]
    (direction_dir / "llm_generated.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_glossary(glossary_path: Path) -> None:
    glossary_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "abbreviation": "IFV",
                        "full_name": "infantry fighting vehicle",
                        "zh": "步兵戰車",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_cli_real_run_trains_records_with_summary_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    generated_root = tmp_path / "generated"
    _write_generated_records(generated_root)
    monkeypatch.setattr(finetune_bootstrap, "_GENERATED_ROOT", generated_root)
    output_dir = tmp_path / "out"

    # Act
    exit_code = finetune_cli.main(
        [
            "-s",
            "en",
            "-t",
            "zh",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    # Assert
    summary = capsys.readouterr().out.strip()
    assert exit_code == 0
    expected_pattern = (
        rf"fine-tuned {_RECORD_COUNT} pairs, "
        r"train_loss=\d+\.\d{4}, saved to " + re.escape(str(output_dir))
    )
    assert re.fullmatch(expected_pattern, summary)
    assert (output_dir / "config.json").is_file()


def test_cli_real_run_with_holdout_prints_three_stdout_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    generated_root = tmp_path / "generated"
    _write_generated_records(generated_root)
    glossary_path = tmp_path / "glossary.json"
    _write_glossary(glossary_path)
    monkeypatch.setattr(finetune_bootstrap, "_GENERATED_ROOT", generated_root)
    monkeypatch.setattr(finetune_bootstrap, "_GLOSSARY_PATH", glossary_path)
    holdout_count = int(0.25 * _RECORD_COUNT + 0.5)

    # Act
    exit_code = finetune_cli.main(
        [
            "-s",
            "en",
            "-t",
            "zh",
            "--holdout-fraction",
            "0.25",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    # Assert
    summary_line, score_line, terms_line = (
        capsys.readouterr().out.strip().splitlines()
    )
    assert exit_code == 0
    assert summary_line.startswith(
        f"fine-tuned {_RECORD_COUNT - holdout_count} pairs, "
    )
    assert re.fullmatch(
        rf"held-out {holdout_count} pairs: "
        r"chrF base=\d+\.\d{2}, tuned=\d+\.\d{2}",
        score_line,
    )
    assert re.fullmatch(r"terms: base=\d+/\d+, tuned=\d+/\d+", terms_line)


def test_cli_real_run_missing_direction_folder_returns_one_with_stderr_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange: only en_to_zh/ exists, so zh -> en has no generated-data folder.
    generated_root = tmp_path / "generated"
    _write_generated_records(generated_root)
    monkeypatch.setattr(finetune_bootstrap, "_GENERATED_ROOT", generated_root)
    output_dir = tmp_path / "out"

    # Act
    exit_code = finetune_cli.main(
        [
            "-s",
            "zh",
            "-t",
            "en",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    # Assert
    assert exit_code == 1
    assert capsys.readouterr().err.startswith("No training pairs remain")
    assert not output_dir.exists()
