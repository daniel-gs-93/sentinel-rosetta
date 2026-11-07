"""Resource-aware: wires the fine-tuning use case to its adapters."""

from pathlib import Path

from sentinel_rosetta.application.finetune_use_case import FineTuneModelUseCase
from sentinel_rosetta.domain.models import LanguageMap
from sentinel_rosetta.infrastructure.hf_pair_translator import HfPairTranslator
from sentinel_rosetta.infrastructure.hf_seq2seq_trainer import (
    HfSeq2SeqTrainer,
)
from sentinel_rosetta.infrastructure.json_glossary_repository import (
    JsonGlossaryRepository,
)
from sentinel_rosetta.infrastructure.jsonl_generated_record_repository import (
    JsonlGeneratedRecordRepository,
)
from sentinel_rosetta.infrastructure.opencc_adapter import OpenCcAdapter

_GENERATED_ROOT = Path("data/llm_generated")
_GLOSSARY_PATH = Path("data/llm_reference/glossary_en_to_zh.json")


def build_finetune_model_use_case(
    source_lang: LanguageMap,
    target_lang: LanguageMap,
    base_model_dir: Path,
    output_dir: Path,
    max_length: int,
) -> FineTuneModelUseCase:
    direction_dir = (
        _GENERATED_ROOT / f"{source_lang.value}_to_{target_lang.value}"
    )
    return FineTuneModelUseCase(
        JsonlGeneratedRecordRepository(direction_dir),
        JsonGlossaryRepository(_GLOSSARY_PATH),
        OpenCcAdapter(),
        HfSeq2SeqTrainer(base_model_dir, output_dir),
        HfPairTranslator(base_model_dir, max_length),
        HfPairTranslator(output_dir, max_length),
    )
