import re
from pathlib import Path
from typing import Optional

import pytest

from sentinel_rosetta.domain.interfaces import TrainingEnvironmentError
from sentinel_rosetta.infrastructure.hf_pair_translator import HfPairTranslator


def _all_packages_installed(package_name: str) -> bool:
    return True


def _torch_missing(package_name: str) -> bool:
    return package_name != "torch"


@pytest.mark.parametrize(
    ("has_config", "weight_file_name"),
    [(False, None), (True, None), (False, "pytorch_model.bin")],
    ids=["nothing", "config-only", "weights-only"],
)
def test_translate_incomplete_model_dir_raises_error_with_directory(
    tmp_path: Path, has_config: bool, weight_file_name: Optional[str]
) -> None:
    # Arrange
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if has_config:
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
    if weight_file_name is not None:
        (model_dir / weight_file_name).write_bytes(b"")
    translator = HfPairTranslator(model_dir, 64, _all_packages_installed)

    # Act / Assert
    with pytest.raises(
        TrainingEnvironmentError, match=re.escape(str(model_dir))
    ):
        translator.translate(("Hello",))


def test_translate_missing_torch_raises_training_environment_error(
    tmp_path: Path,
) -> None:
    # Arrange
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"")
    translator = HfPairTranslator(model_dir, 64, _torch_missing)

    # Act / Assert
    with pytest.raises(TrainingEnvironmentError, match="torch"):
        translator.translate(("Hello",))
