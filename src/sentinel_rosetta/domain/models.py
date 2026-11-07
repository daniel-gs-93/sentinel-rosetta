"""Pure logic: immutable data types shared across layers."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class LanguageMap(str, Enum):
    ENGLISH = "en"
    MANDARIN = "zh"


@dataclass(frozen=True)
class TranslationResult:
    source_text: str
    target_text: str
    source_lang: LanguageMap
    target_lang: LanguageMap


@dataclass(frozen=True)
class GeneratedRecord:
    id: str
    zh: str
    en: str
    origin: str


@dataclass(frozen=True)
class GlossaryEntry:
    abbreviation: Optional[str]
    full_name: str
    zh: str


@dataclass(frozen=True)
class TrainingPair:
    source_text: str
    target_text: str


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    seed: int


@dataclass(frozen=True)
class EvaluationRow:
    source_text: str
    reference_text: str
    base_output: str
    tuned_output: str
    base_score: float
    tuned_score: float
    expected_term_count: int
    base_hit_count: int
    tuned_hit_count: int


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    base_score: float
    tuned_score: float
    expected_term_count: int
    base_hit_count: int
    tuned_hit_count: int
    rows: tuple[EvaluationRow, ...]


@dataclass(frozen=True)
class TrainingResult:
    pair_count: int
    train_loss: float
    output_dir: Path
    evaluation: Optional[EvaluationReport] = None
