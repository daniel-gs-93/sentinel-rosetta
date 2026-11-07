"""Pure logic: hold-out split and training pair construction."""

import functools
import hashlib

from sentinel_rosetta.domain.interfaces import ScriptConverter
from sentinel_rosetta.domain.models import (
    GeneratedRecord,
    LanguageMap,
    TrainingPair,
)


def _holdout_rank(seed: int, record: GeneratedRecord) -> str:
    return hashlib.sha256(f"{seed}:{record.id}".encode()).hexdigest()


def split_holdout(
    records: tuple[GeneratedRecord, ...], holdout_fraction: float, seed: int
) -> tuple[tuple[GeneratedRecord, ...], tuple[GeneratedRecord, ...]]:
    holdout_size = int(holdout_fraction * len(records) + 0.5)
    ranked_records = sorted(records, key=functools.partial(_holdout_rank, seed))
    holdout_ids = frozenset(
        record.id for record in ranked_records[:holdout_size]
    )
    train_records = tuple(
        record for record in records if record.id not in holdout_ids
    )
    holdout_records = tuple(
        record for record in records if record.id in holdout_ids
    )
    return train_records, holdout_records


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_training_pairs(
    records: tuple[GeneratedRecord, ...],
    source_lang: LanguageMap,
    target_lang: LanguageMap,
    script_converter: ScriptConverter,
) -> tuple[TrainingPair, ...]:
    if source_lang == target_lang:
        raise ValueError("source_lang and target_lang must differ")

    training_pairs: list[TrainingPair] = []
    for record in records:
        english_text = _normalize_whitespace(record.en)
        chinese_text = _normalize_whitespace(record.zh)
        if not english_text or not chinese_text:
            continue
        simplified_text = script_converter.to_simplified(chinese_text)
        if source_lang == LanguageMap.ENGLISH:
            training_pairs.append(TrainingPair(english_text, simplified_text))
        else:
            training_pairs.append(TrainingPair(simplified_text, english_text))
    return tuple(training_pairs)
