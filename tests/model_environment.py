from pathlib import Path

MODELS_DIR = Path("models")
_REQUIRED_DIRS = (
    "opus-mt-en-zh",
    "opus-mt-en-zh-ct2",
    "opus-mt-zh-en",
    "opus-mt-zh-en-ct2",
)


def missing_translation_models() -> list[str]:
    """List the model directories an Integration or Environment test needs."""
    return [
        str(MODELS_DIR / name)
        for name in _REQUIRED_DIRS
        if not (MODELS_DIR / name).is_dir()
    ]
