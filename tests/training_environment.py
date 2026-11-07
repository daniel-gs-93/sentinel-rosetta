import importlib.util
from pathlib import Path

BASE_MODEL_DIR = Path("models/hf/opus-mt-en-zh")


def missing_training_prerequisites() -> list[str]:
    """List what an Environment-tier fine-tuning test still needs."""
    missing = [
        f"package {package_name}"
        for package_name in ("torch", "accelerate")
        if importlib.util.find_spec(package_name) is None
    ]
    if not (BASE_MODEL_DIR / "config.json").is_file():
        missing.append(f"checkpoint {BASE_MODEL_DIR}")
    return missing
