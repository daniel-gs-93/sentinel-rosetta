import math
from pathlib import Path

import pytest

from sentinel_rosetta.domain.models import TrainingConfig, TrainingPair
from sentinel_rosetta.infrastructure.hf_seq2seq_trainer import HfSeq2SeqTrainer
from tests.training_environment import (
    BASE_MODEL_DIR,
    missing_training_prerequisites,
)

_MISSING = missing_training_prerequisites()

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"fine-tuning prerequisites missing: {_MISSING}",
)


def test_train_real_checkpoint_writes_reloadable_checkpoint(
    tmp_path: Path,
) -> None:
    # Arrange
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output_dir = tmp_path / "out"
    pairs = (
        TrainingPair("The UAV returned to base.", "无人飞行载具返回基地。"),
        TrainingPair("The MBT crossed the river.", "主战坦克越过了河。"),
        TrainingPair("Today's weather is great.", "今天天气好。"),
    )
    config = TrainingConfig(
        epochs=1, batch_size=2, learning_rate=5e-5, max_length=64, seed=1
    )

    # Act
    result = HfSeq2SeqTrainer(BASE_MODEL_DIR, output_dir).train(pairs, config)

    # Assert
    assert result.pair_count == 3
    assert math.isfinite(result.train_loss) and result.train_loss > 0
    assert (output_dir / "config.json").is_file()
    assert AutoModelForSeq2SeqLM.from_pretrained(str(output_dir)) is not None
    # transformers ships no type hints for from_pretrained.
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        str(output_dir)
    )
    assert tokenizer is not None
