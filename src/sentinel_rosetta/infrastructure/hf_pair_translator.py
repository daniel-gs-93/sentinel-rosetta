"""Resource-aware: translates sentences with a Hugging Face checkpoint."""

from collections.abc import Callable
from pathlib import Path

from sentinel_rosetta.domain.interfaces import TrainingEnvironmentError
from sentinel_rosetta.infrastructure._hf_checkpoint import (
    WEIGHT_FILE_NAMES,
    has_checkpoint_files,
    is_package_installed,
)

# torch is imported by transformers only when a model is loaded, so importing
# this module works without it (ADR-0006 in docs/DecisionRecord.md).
_REQUIRED_PACKAGES = ("torch",)


class HfPairTranslator:
    def __init__(
        self,
        model_dir: Path,
        max_length: int,
        is_package_installed: Callable[[str], bool] = is_package_installed,
        batch_size: int = 8,
    ) -> None:
        self._model_dir = model_dir
        self._max_length = max_length
        self._is_package_installed = is_package_installed
        self._batch_size = batch_size

    def translate(self, source_texts: tuple[str, ...]) -> tuple[str, ...]:
        self._check_packages_are_installed()
        self._check_checkpoint_exists()
        return self._generate(source_texts)

    def _check_packages_are_installed(self) -> None:
        missing_packages = [
            package_name
            for package_name in _REQUIRED_PACKAGES
            if not self._is_package_installed(package_name)
        ]
        if missing_packages:
            raise TrainingEnvironmentError(
                f"Missing packages for evaluation: {missing_packages}. "
                "Install them on the training machine (see README.md, "
                "Fine-tuning)."
            )

    def _check_checkpoint_exists(self) -> None:
        if not has_checkpoint_files(self._model_dir):
            raise TrainingEnvironmentError(
                f"Model directory {self._model_dir} must contain "
                f"config.json and one of {list(WEIGHT_FILE_NAMES)}."
            )

    def _generate(self, source_texts: tuple[str, ...]) -> tuple[str, ...]:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            str(self._model_dir)
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self._model_dir))

        translations: list[str] = []
        for start in range(0, len(source_texts), self._batch_size):
            batch = list(source_texts[start : start + self._batch_size])
            encoded_batch = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            generated_ids = model.generate(
                **encoded_batch, max_new_tokens=self._max_length
            )
            translations.extend(
                tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )
        return tuple(translations)
