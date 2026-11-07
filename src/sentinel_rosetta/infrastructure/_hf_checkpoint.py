"""Resource-aware: probes installed packages and checkpoint files."""

import importlib.util
from pathlib import Path

WEIGHT_FILE_NAMES = ("model.safetensors", "pytorch_model.bin")


def is_package_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def has_checkpoint_files(model_dir: Path) -> bool:
    has_config = (model_dir / "config.json").is_file()
    has_weights = any(
        (model_dir / weight_file_name).is_file()
        for weight_file_name in WEIGHT_FILE_NAMES
    )
    return has_config and has_weights
