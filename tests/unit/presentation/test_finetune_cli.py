from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel_rosetta.application.finetune_use_case import FineTuneModelUseCase
from sentinel_rosetta.domain.interfaces import (
    GeneratedRecordFormatError,
    GlossaryFormatError,
    NoEvaluationDataError,
    NoTrainingDataError,
    TrainingEnvironmentError,
)
from sentinel_rosetta.domain.models import (
    EvaluationReport,
    EvaluationRow,
    LanguageMap,
    TrainingConfig,
    TrainingResult,
)
from sentinel_rosetta.presentation import finetune_cli


@pytest.fixture
def use_case_factory(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    use_case = MagicMock(spec=FineTuneModelUseCase)
    use_case.execute.return_value = TrainingResult(
        pair_count=3, train_loss=0.12345, output_dir=Path("out/dir")
    )
    factory = MagicMock(return_value=use_case)
    monkeypatch.setattr(finetune_cli, "build_finetune_model_use_case", factory)
    return factory


def test_main_valid_arguments_returns_zero_with_summary_on_stdout(
    capsys: pytest.CaptureFixture[str], use_case_factory: MagicMock
) -> None:
    # Act
    exit_code = finetune_cli.main(["-s", "en", "-t", "zh"])

    # Assert
    assert exit_code == 0
    assert (
        capsys.readouterr().out.strip()
        == "fine-tuned 3 pairs, train_loss=0.1235, saved to out/dir"
    )


def test_main_passes_parsed_arguments_to_use_case(
    use_case_factory: MagicMock,
) -> None:
    # Act
    finetune_cli.main(
        [
            "-s",
            "zh",
            "-t",
            "en",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.001",
            "--holdout-fraction",
            "0.25",
            "--base-model-dir",
            "my/base",
            "--output-dir",
            "my/out",
        ]
    )

    # Assert
    use_case_factory.assert_called_once_with(
        LanguageMap.MANDARIN,
        LanguageMap.ENGLISH,
        Path("my/base"),
        Path("my/out"),
        128,
    )
    use_case_factory.return_value.execute.assert_called_once_with(
        LanguageMap.MANDARIN,
        LanguageMap.ENGLISH,
        TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=0.001,
            max_length=128,
            seed=42,
        ),
        0.25,
    )


def test_main_source_equals_target_returns_one_with_message_on_stderr(
    capsys: pytest.CaptureFixture[str], use_case_factory: MagicMock
) -> None:
    # Act
    exit_code = finetune_cli.main(["-s", "en", "-t", "en"])

    # Assert
    assert exit_code == 1
    assert capsys.readouterr().err != ""
    use_case_factory.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        NoTrainingDataError("no pairs left"),
        GeneratedRecordFormatError("bad record line"),
        GlossaryFormatError("bad glossary entry"),
        NoEvaluationDataError("hold-out is empty"),
        TrainingEnvironmentError("torch missing"),
    ],
    ids=[
        "no-training-data",
        "record-format",
        "glossary-format",
        "no-evaluation-data",
        "training-environment",
    ],
)
def test_main_domain_failure_returns_one_with_message_on_stderr(
    capsys: pytest.CaptureFixture[str],
    use_case_factory: MagicMock,
    error: Exception,
) -> None:
    # Arrange
    use_case_factory.return_value.execute.side_effect = error

    # Act
    exit_code = finetune_cli.main(["-s", "en", "-t", "zh"])

    # Assert
    assert exit_code == 1
    assert str(error) in capsys.readouterr().err


def test_main_missing_required_argument_exits_with_code_two() -> None:
    # Act / Assert
    with pytest.raises(SystemExit) as exit_info:
        finetune_cli.main(["-s", "en"])
    assert exit_info.value.code == 2


def test_main_removed_include_machine_generated_flag_exits_with_code_two() -> (
    None
):
    # Act / Assert
    with pytest.raises(SystemExit) as exit_info:
        finetune_cli.main(
            ["-s", "en", "-t", "zh", "--include-machine-generated"]
        )
    assert exit_info.value.code == 2


def test_main_holdout_prints_evaluation_report(
    capsys: pytest.CaptureFixture[str], use_case_factory: MagicMock
) -> None:
    # Arrange
    report = EvaluationReport(
        sample_count=2,
        base_score=12.3456,
        tuned_score=56.7891,
        expected_term_count=3,
        base_hit_count=1,
        tuned_hit_count=2,
        rows=(
            EvaluationRow(
                "src one",
                "ref one",
                "base one",
                "tuned one",
                10.0,
                50.0,
                1,
                0,
                1,
            ),
            EvaluationRow(
                "src two",
                "ref two",
                "base two",
                "tuned two",
                14.0,
                60.0,
                2,
                1,
                1,
            ),
        ),
    )
    use_case_factory.return_value.execute.return_value = TrainingResult(
        pair_count=6,
        train_loss=1.0,
        output_dir=Path("out/dir"),
        evaluation=report,
    )

    # Act
    exit_code = finetune_cli.main(
        ["-s", "en", "-t", "zh", "--holdout-fraction", "0.25"]
    )

    # Assert
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().splitlines() == [
        "fine-tuned 6 pairs, train_loss=1.0000, saved to out/dir",
        "held-out 2 pairs: chrF base=12.35, tuned=56.79",
        "terms: base=1/3, tuned=2/3",
    ]
    for expected in ("src two", "ref two", "base two", "tuned two"):
        assert expected in captured.err
