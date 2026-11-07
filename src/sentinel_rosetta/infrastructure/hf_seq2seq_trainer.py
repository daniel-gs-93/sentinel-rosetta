"""Resource-aware: fine-tunes a Hugging Face checkpoint and saves it."""

import contextlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from sentinel_rosetta.domain.interfaces import TrainingEnvironmentError
from sentinel_rosetta.domain.models import (
    TrainingConfig,
    TrainingPair,
    TrainingResult,
)
from sentinel_rosetta.infrastructure._hf_checkpoint import (
    WEIGHT_FILE_NAMES,
    has_checkpoint_files,
    is_package_installed,
)

# PyTorch and accelerate are installed only on the training machine and are
# not declared in pyproject.toml (ADR-0006 in docs/DecisionRecord.md).
# They are imported inside _run_training so that importing this module, and
# every test that does not train, works without them.
_REQUIRED_PACKAGES = ("torch", "accelerate")

_logger = logging.getLogger(__name__)


class HfSeq2SeqTrainer:
    def __init__(
        self,
        base_model_dir: Path,
        output_dir: Path,
        is_package_installed: Callable[[str], bool] = is_package_installed,
    ) -> None:
        self._base_model_dir = base_model_dir
        self._output_dir = output_dir
        self._is_package_installed = is_package_installed

    def train(
        self, pairs: tuple[TrainingPair, ...], config: TrainingConfig
    ) -> TrainingResult:
        self._check_output_dir_is_free()
        self._check_packages_are_installed()
        self._check_base_model_exists()
        return self._run_training(pairs, config)

    def _check_output_dir_is_free(self) -> None:
        if not self._output_dir.exists():
            return
        if self._output_dir.is_dir() and not any(self._output_dir.iterdir()):
            return
        raise TrainingEnvironmentError(
            f"Output directory {self._output_dir} is not an empty "
            "directory; remove it or choose another --output-dir."
        )

    def _check_packages_are_installed(self) -> None:
        missing_packages = [
            package_name
            for package_name in _REQUIRED_PACKAGES
            if not self._is_package_installed(package_name)
        ]
        if missing_packages:
            raise TrainingEnvironmentError(
                f"Missing packages for fine-tuning: {missing_packages}. "
                "Install them on the training machine (see README.md, "
                "Fine-tuning)."
            )

    def _check_base_model_exists(self) -> None:
        if not has_checkpoint_files(self._base_model_dir):
            raise TrainingEnvironmentError(
                f"Base model directory {self._base_model_dir} must contain "
                f"config.json and one of {list(WEIGHT_FILE_NAMES)} "
                "(see README.md, Fine-tuning)."
            )

    def _run_training(
        self, pairs: tuple[TrainingPair, ...], config: TrainingConfig
    ) -> TrainingResult:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            str(self._base_model_dir)
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self._base_model_dir))

        encodings = tokenizer(
            [pair.source_text for pair in pairs],
            text_target=[pair.target_text for pair in pairs],
            max_length=config.max_length,
            truncation=True,
        )
        train_features = [
            {
                "input_ids": encodings["input_ids"][pair_index],
                "attention_mask": encodings["attention_mask"][pair_index],
                "labels": encodings["labels"][pair_index],
            }
            for pair_index in range(len(pairs))
        ]

        training_arguments = Seq2SeqTrainingArguments(
            output_dir=str(self._output_dir),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
            logging_steps=10,
            remove_unused_columns=False,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_features,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer,
        )
        _logger.info(
            f"Fine-tuning {self._base_model_dir} on {len(pairs)} pairs "
            f"for {config.epochs} epochs"
        )
        # Seq2SeqTrainer prints its metrics to stdout; the CLI reserves
        # stdout for its one-line summary, so training output goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            train_output = trainer.train()

        trainer.save_model(str(self._output_dir))
        tokenizer.save_pretrained(str(self._output_dir))
        return TrainingResult(
            pair_count=len(pairs),
            train_loss=float(train_output.training_loss),
            output_dir=self._output_dir,
        )
