import pytest

from sentinel_rosetta.infrastructure.hf_pair_translator import HfPairTranslator
from tests.training_environment import (
    BASE_MODEL_DIR,
    missing_training_prerequisites,
)

_MISSING = missing_training_prerequisites()

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"fine-tuning prerequisites missing: {_MISSING}",
)


def test_translate_real_checkpoint_returns_one_output_per_source() -> None:
    # Arrange
    sources = (
        "The UAV returned to base.",
        "The MBT crossed the river.",
        "Each system has inherent strengths and weaknesses.",
    )
    translator = HfPairTranslator(BASE_MODEL_DIR, 64)

    # Act
    outputs = translator.translate(sources)

    # Assert
    assert len(outputs) == 3
    assert all(output.strip() != "" for output in outputs)
