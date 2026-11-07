import shutil
from pathlib import Path

import pytest

from sentinel_rosetta.domain.models import (
    LanguageMap,
    TrainingConfig,
    TrainingPair,
)
from sentinel_rosetta.infrastructure.ctranslate2_adapter import (
    CTranslate2Adapter,
)
from sentinel_rosetta.infrastructure.hf_seq2seq_trainer import HfSeq2SeqTrainer
from tests.model_environment import MODELS_DIR, missing_translation_models
from tests.training_environment import (
    BASE_MODEL_DIR,
    missing_training_prerequisites,
)

_MISSING = missing_training_prerequisites() + missing_translation_models()

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"conversion prerequisites missing: {_MISSING}",
)


def test_converted_checkpoint_is_translatable_by_the_engine(
    tmp_path: Path,
) -> None:
    # Arrange: train a tiny checkpoint, then convert and install it into a
    # models directory laid out like the translation service expects.
    from ctranslate2.converters import TransformersConverter

    checkpoint_dir = tmp_path / "checkpoint"
    pairs = (
        TrainingPair("The UAV returned to base.", "无人飞行载具返回基地。"),
        TrainingPair("The MBT crossed the river.", "主战坦克越过了河。"),
        TrainingPair("Today's weather is great.", "今天天气好。"),
    )
    config = TrainingConfig(
        epochs=1, batch_size=2, learning_rate=5e-5, max_length=64, seed=1
    )
    HfSeq2SeqTrainer(BASE_MODEL_DIR, checkpoint_dir).train(pairs, config)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    converted_dir = models_dir / "opus-mt-en-zh-ct2"

    # Act
    TransformersConverter(str(checkpoint_dir)).convert(
        str(converted_dir), quantization="int8"
    )
    shutil.copytree(
        checkpoint_dir,
        models_dir / "opus-mt-en-zh",
        ignore=shutil.ignore_patterns(
            "model.safetensors", "pytorch_model.bin", "training_args.bin"
        ),
    )
    for name in ("opus-mt-zh-en", "opus-mt-zh-en-ct2"):
        shutil.copytree(MODELS_DIR / name, models_dir / name)
    result = CTranslate2Adapter(models_dir=models_dir).translate(
        "Hello", LanguageMap.ENGLISH, LanguageMap.MANDARIN
    )

    # Assert
    converted_names = {path.name for path in converted_dir.iterdir()}
    assert {"config.json", "model.bin", "shared_vocabulary.json"} <= (
        converted_names
    )
    assert result.target_text.strip() != ""
