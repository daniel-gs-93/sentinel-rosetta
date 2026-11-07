import inspect

import pytest

from sentinel_rosetta.application.use_cases import TranslateTextUseCase
from sentinel_rosetta.domain.interfaces import TranslationInputTooLongError
from sentinel_rosetta.domain.models import LanguageMap
from tests.doubles import FakeScriptConverter, FakeTranslationEngine

_EN = LanguageMap.ENGLISH
_ZH = LanguageMap.MANDARIN


def test_execute_en_to_zh_sends_original_text_to_engine() -> None:
    # Arrange
    engine = FakeTranslationEngine()
    use_case = TranslateTextUseCase(engine, FakeScriptConverter())

    # Act
    use_case.execute("The IFV arrived.", _EN, _ZH)

    # Assert
    assert engine.calls == [("The IFV arrived.", _EN, _ZH)]


def test_execute_en_to_zh_converts_engine_output_to_traditional() -> None:
    # Arrange: the fake converter turns the Simplified marker 简 into 繁.
    engine = FakeTranslationEngine(target_text="简")
    use_case = TranslateTextUseCase(engine, FakeScriptConverter())

    # Act
    result = use_case.execute("Hello", _EN, _ZH)

    # Assert
    assert result.target_text == "繁"


def test_execute_en_to_zh_converts_script_only_after_translating() -> None:
    # Arrange
    call_log: list[str] = []
    engine = FakeTranslationEngine(call_log=call_log)
    use_case = TranslateTextUseCase(
        engine, FakeScriptConverter(call_log=call_log)
    )

    # Act
    use_case.execute("Hello", _EN, _ZH)

    # Assert
    assert call_log == ["engine.translate", "converter.to_traditional"]


def test_execute_zh_to_en_translates_the_simplified_input() -> None:
    # Arrange
    engine = FakeTranslationEngine()
    use_case = TranslateTextUseCase(engine, FakeScriptConverter())

    # Act
    use_case.execute("繁", _ZH, _EN)

    # Assert
    assert engine.calls == [("简", _ZH, _EN)]


def test_execute_zh_to_en_returns_engine_output_without_script_conversion() -> (
    None
):
    # Arrange
    call_log: list[str] = []
    engine = FakeTranslationEngine(
        target_text="The IFV arrived.", call_log=call_log
    )
    use_case = TranslateTextUseCase(
        engine, FakeScriptConverter(call_log=call_log)
    )

    # Act
    result = use_case.execute("繁", _ZH, _EN)

    # Assert
    assert result.target_text == "The IFV arrived."
    assert "converter.to_traditional" not in call_log


@pytest.mark.parametrize(
    ("text", "source_lang", "target_lang"),
    [("Hello", _EN, _ZH), ("繁", _ZH, _EN)],
    ids=["en-to-zh", "zh-to-en"],
)
def test_execute_source_text_is_the_original_input(
    text: str, source_lang: LanguageMap, target_lang: LanguageMap
) -> None:
    # Arrange
    use_case = TranslateTextUseCase(
        FakeTranslationEngine(), FakeScriptConverter()
    )

    # Act
    result = use_case.execute(text, source_lang, target_lang)

    # Assert
    assert result.source_text == text


@pytest.mark.parametrize(
    ("source_lang", "target_lang"),
    [(_EN, _ZH), (_ZH, _EN)],
    ids=["en-to-zh", "zh-to-en"],
)
def test_execute_calls_translate_exactly_once(
    source_lang: LanguageMap, target_lang: LanguageMap
) -> None:
    # Arrange
    engine = FakeTranslationEngine()
    use_case = TranslateTextUseCase(engine, FakeScriptConverter())

    # Act
    use_case.execute("Hello", source_lang, target_lang)

    # Assert
    assert len(engine.calls) == 1


def test_execute_engine_error_propagates_unchanged() -> None:
    # Arrange
    error = TranslationInputTooLongError("too long")
    use_case = TranslateTextUseCase(
        FakeTranslationEngine(error=error), FakeScriptConverter()
    )

    # Act / Assert
    with pytest.raises(TranslationInputTooLongError) as error_info:
        use_case.execute("Hello", _EN, _ZH)
    assert error_info.value is error


def test_use_case_constructor_parameters_are_engine_then_converter() -> None:
    # Act
    parameter_names = list(inspect.signature(TranslateTextUseCase).parameters)

    # Assert
    assert parameter_names == ["translation_engine", "script_converter"]
