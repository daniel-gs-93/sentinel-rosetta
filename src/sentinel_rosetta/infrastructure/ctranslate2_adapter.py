"""Resource-aware: runs CTranslate2 models loaded from disk."""

import logging
from pathlib import Path

from sentinel_rosetta.infrastructure import _offline_env  # isort: skip  # noqa: F401

import ctranslate2
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sentinel_rosetta.domain.interfaces import TranslationInputTooLongError
from sentinel_rosetta.domain.models import LanguageMap, TranslationResult

# Opus-MT/Marian's position embedding table only covers 512 positions;
# CTranslate2 raises a bare RuntimeError past that, not a validation error.
_MAX_SOURCE_TOKENS = 512

_MODEL_DIRS = {
    (LanguageMap.ENGLISH, LanguageMap.MANDARIN): "opus-mt-en-zh",
    (LanguageMap.MANDARIN, LanguageMap.ENGLISH): "opus-mt-zh-en",
}


class CTranslate2Adapter:
    def __init__(self, models_dir: Path) -> None:
        self._logger = logging.getLogger(__name__)
        self._tokenizers: dict[
            tuple[LanguageMap, LanguageMap], PreTrainedTokenizerBase
        ] = {}
        self._translators: dict[
            tuple[LanguageMap, LanguageMap], ctranslate2.Translator
        ] = {}

        for direction, model_name in _MODEL_DIRS.items():
            tokenizer_dir = models_dir / model_name
            translator_dir = models_dir / f"{model_name}-ct2"
            # from_pretrained treats an absolute path that does not exist as
            # a malformed Hub repo id and raises ValueError, not OSError, so
            # a missing directory is checked here first.
            if not tokenizer_dir.is_dir() or not translator_dir.is_dir():
                raise RuntimeError(f"Model binaries missing for {direction}.")
            try:
                self._tokenizers[direction] = AutoTokenizer.from_pretrained(
                    str(tokenizer_dir)
                )
                self._translators[direction] = ctranslate2.Translator(
                    str(translator_dir), device="cpu"
                )
            except (OSError, RuntimeError) as e:
                raise RuntimeError(
                    f"Model binaries missing for {direction}."
                ) from e

        self._logger.info("CTranslate2 pipelines ready.")

    def translate(
        self, text: str, source_lang: LanguageMap, target_lang: LanguageMap
    ) -> TranslationResult:
        direction = (source_lang, target_lang)
        tokenizer = self._tokenizers[direction]
        translator = self._translators[direction]

        source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        if len(source_tokens) > _MAX_SOURCE_TOKENS:
            raise TranslationInputTooLongError(
                f"Input tokenizes to {len(source_tokens)} tokens, "
                f"exceeding the model's {_MAX_SOURCE_TOKENS}-token limit."
            )

        results = translator.translate_batch([source_tokens])
        target_tokens = results[0].hypotheses[0]
        target_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(target_tokens),
            skip_special_tokens=True,
        )

        return TranslationResult(
            source_text=text,
            target_text=target_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
