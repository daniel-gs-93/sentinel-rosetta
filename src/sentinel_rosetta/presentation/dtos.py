"""Pure logic: validated request types for the API and the CLIs."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from sentinel_rosetta.domain.models import LanguageMap, TrainingConfig

_DEFAULT_BASE_MODEL_ROOT = Path("models/hf")
_DEFAULT_OUTPUT_ROOT = Path("models/finetuned")


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: LanguageMap
    target_lang: LanguageMap

    @model_validator(mode="after")
    def check_languages_differ(self) -> "TranslationRequest":
        if self.source_lang == self.target_lang:
            raise ValueError("source_lang and target_lang must differ")
        return self


class FinetuneRequest(BaseModel):
    source_lang: LanguageMap
    target_lang: LanguageMap
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    holdout_fraction: float = Field(default=0.0, ge=0, lt=1)
    max_length: int = Field(default=128, ge=1)
    seed: int = 42
    base_model_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    @model_validator(mode="after")
    def check_languages_differ(self) -> "FinetuneRequest":
        if self.source_lang == self.target_lang:
            raise ValueError("source_lang and target_lang must differ")
        return self

    def resolved_base_model_dir(self) -> Path:
        return (
            self.base_model_dir or _DEFAULT_BASE_MODEL_ROOT / self._model_name()
        )

    def resolved_output_dir(self) -> Path:
        return self.output_dir or _DEFAULT_OUTPUT_ROOT / self._model_name()

    def to_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_length=self.max_length,
            seed=self.seed,
        )

    def _model_name(self) -> str:
        return f"opus-mt-{self.source_lang.value}-{self.target_lang.value}"
