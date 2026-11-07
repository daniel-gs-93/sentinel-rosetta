"""Resource-aware: downloads the Opus-MT models and converts them."""

import logging
from pathlib import Path

from ctranslate2.converters import TransformersConverter
from transformers import AutoTokenizer

# PyTorch and Transformers are required only for this one-time step.
# TransformersConverter resolves a Hub repo id itself (downloads and loads
# the original tensors internally), so this script never calls
# AutoModelForSeq2SeqLM directly. ctranslate2_adapter.py reads the tokenizer
# from the plain directory below and the model from *-ct2/, without PyTorch.
_MODELS_DIR = Path("models")
_HF_REPOS = ["Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"]

_logger = logging.getLogger(__name__)


def _download_and_convert(hf_repo: str) -> None:
    repo_name = hf_repo.split("/")[-1]
    tokenizer_dir = _MODELS_DIR / repo_name
    ct2_dir = _MODELS_DIR / f"{repo_name}-ct2"

    if tokenizer_dir.exists():
        _logger.info(f"{tokenizer_dir} already downloaded")
    else:
        _logger.info(f"Downloading tokenizer for {hf_repo} to {tokenizer_dir}")
        AutoTokenizer.from_pretrained(hf_repo).save_pretrained(tokenizer_dir)

    if ct2_dir.exists():
        _logger.info(f"{ct2_dir} already converted")
    else:
        _logger.info(f"Converting {hf_repo} to {ct2_dir}")
        TransformersConverter(hf_repo).convert(
            str(ct2_dir), quantization="int8"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for hf_repo in _HF_REPOS:
        _download_and_convert(hf_repo)


if __name__ == "__main__":
    main()
