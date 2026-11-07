import re
from pathlib import Path
from typing import Optional

import pytest

from sentinel_rosetta.domain.interfaces import TrainingEnvironmentError
from sentinel_rosetta.domain.models import TrainingConfig, TrainingPair
from sentinel_rosetta.infrastructure.hf_seq2seq_trainer import HfSeq2SeqTrainer

_CONFIG = TrainingConfig(
    epochs=1, batch_size=2, learning_rate=1e-4, max_length=64, seed=1
)
_PAIRS = (TrainingPair("Hello", "你好"),)


class FakePackageFinder:
    """A package-presence check that reports a fixed set of packages missing."""

    def __init__(self, missing_packages: frozenset[str] = frozenset()) -> None:
        self._missing_packages = missing_packages

    def __call__(self, package_name: str) -> bool:
        return package_name not in self._missing_packages


def _trainer(
    base_model_dir: Path,
    output_dir: Path,
    missing_packages: frozenset[str] = frozenset(),
) -> HfSeq2SeqTrainer:
    return HfSeq2SeqTrainer(
        base_model_dir, output_dir, FakePackageFinder(missing_packages)
    )


def test_train_output_dir_not_empty_raises_error_without_touching_files(
    tmp_path: Path,
) -> None:
    # Arrange
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_file = output_dir / "keep.txt"
    existing_file.write_text("precious", encoding="utf-8")

    # Act / Assert
    with pytest.raises(TrainingEnvironmentError):
        _trainer(tmp_path / "base", output_dir).train(_PAIRS, _CONFIG)
    assert existing_file.read_text(encoding="utf-8") == "precious"


def test_train_empty_output_dir_passes_output_check(tmp_path: Path) -> None:
    # Arrange: both packages are missing, so the error must come from the
    # package check, which only runs after the output check passed.
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    trainer = _trainer(
        tmp_path / "base", output_dir, frozenset({"torch", "accelerate"})
    )

    # Act / Assert
    with pytest.raises(TrainingEnvironmentError, match="Missing packages"):
        trainer.train(_PAIRS, _CONFIG)


@pytest.mark.parametrize(
    "missing_packages",
    [
        frozenset({"torch"}),
        frozenset({"accelerate"}),
        frozenset({"torch", "accelerate"}),
    ],
    ids=["torch", "accelerate", "both"],
)
def test_train_missing_packages_error_names_each_missing_package(
    tmp_path: Path, missing_packages: frozenset[str]
) -> None:
    # Arrange
    trainer = _trainer(tmp_path / "base", tmp_path / "out", missing_packages)

    # Act
    with pytest.raises(TrainingEnvironmentError) as error_info:
        trainer.train(_PAIRS, _CONFIG)

    # Assert
    message = str(error_info.value)
    assert all(package in message for package in missing_packages)
    assert all(
        package not in message
        for package in {"torch", "accelerate"} - missing_packages
    )


@pytest.mark.parametrize(
    ("has_config", "weight_file_name"),
    [(False, None), (True, None), (False, "pytorch_model.bin")],
    ids=["nothing", "config-only", "weights-only"],
)
def test_train_incomplete_base_model_raises_error_with_directory(
    tmp_path: Path, has_config: bool, weight_file_name: Optional[str]
) -> None:
    # Arrange
    base_model_dir = tmp_path / "base"
    base_model_dir.mkdir()
    if has_config:
        (base_model_dir / "config.json").write_text("{}", encoding="utf-8")
    if weight_file_name is not None:
        (base_model_dir / weight_file_name).write_bytes(b"")
    trainer = _trainer(base_model_dir, tmp_path / "out")

    # Act / Assert
    with pytest.raises(
        TrainingEnvironmentError, match=re.escape(str(base_model_dir))
    ):
        trainer.train(_PAIRS, _CONFIG)
