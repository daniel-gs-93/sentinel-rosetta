import pytest

from sentinel_rosetta.domain.glossary_terms import (
    simplify_entries,
    terms_in_text,
)
from sentinel_rosetta.domain.models import GlossaryEntry, LanguageMap
from tests.doubles import FakeScriptConverter, make_entry

_IFV = make_entry("IFV", "infantry fighting vehicle", "步兵战车")
_APC = make_entry("APC", "armored personnel carrier", "装甲运兵车")


def test_terms_in_text_english_abbreviation_matches_whole_token() -> None:
    # Act
    found = terms_in_text("The IFV arrived.", LanguageMap.ENGLISH, (_IFV,))

    # Assert
    assert found == (_IFV,)


def test_terms_in_text_english_abbreviation_is_case_sensitive() -> None:
    # Act
    found = terms_in_text("The ifv arrived.", LanguageMap.ENGLISH, (_IFV,))

    # Assert
    assert found == ()


def test_terms_in_text_english_abbreviation_allows_plural_suffix() -> None:
    # Act
    found = terms_in_text("Two IFVs arrived.", LanguageMap.ENGLISH, (_IFV,))

    # Assert
    assert found == (_IFV,)


@pytest.mark.parametrize(
    "text",
    ["The CA-BDE moved.", "The XCA moved.", "The CA_1 moved."],
    ids=["hyphen", "letter-prefix", "underscore"],
)
def test_terms_in_text_english_abbreviation_inside_longer_token_does_not_match(
    text: str,
) -> None:
    # Arrange
    entry = make_entry("CA", "heavy cruiser", "重巡洋艦")

    # Act
    found = terms_in_text(text, LanguageMap.ENGLISH, (entry,))

    # Assert
    assert found == ()


def test_terms_in_text_english_multi_word_full_name_matches_ignoring_case() -> (
    None
):
    # Act
    found = terms_in_text(
        "Infantry Fighting Vehicles crossed.", LanguageMap.ENGLISH, (_IFV,)
    )

    # Assert
    assert found == (_IFV,)


def test_terms_in_text_english_single_word_full_name_does_not_match() -> None:
    # Arrange
    entry = make_entry("AP", "transport", "運輸艦")

    # Act
    found = terms_in_text(
        "The transport arrived.", LanguageMap.ENGLISH, (entry,)
    )

    # Assert
    assert found == ()


def test_terms_in_text_entry_without_abbreviation_matches_by_full_name() -> (
    None
):
    # Arrange
    entry = make_entry(None, "course of action", "行動方案")

    # Act
    found = terms_in_text(
        "Each course of action was ranked.", LanguageMap.ENGLISH, (entry,)
    )

    # Assert
    assert found == (entry,)


def test_terms_in_text_mandarin_matches_expected_zh_as_substring() -> None:
    # Act
    found = terms_in_text("步兵战车到达基地。", LanguageMap.MANDARIN, (_IFV,))

    # Assert
    assert found == (_IFV,)


def test_terms_in_text_mandarin_cuts_expected_zh_at_first_parenthesis() -> None:
    # Arrange
    entry = make_entry("AA", "antiaircraft", "防空（高射）")

    # Act
    found = terms_in_text("防空部队就位。", LanguageMap.MANDARIN, (entry,))

    # Assert
    assert found == (entry,)


@pytest.mark.parametrize(
    ("text", "entries"),
    [("", (_IFV,)), ("The IFV arrived.", ())],
    ids=["empty-text", "empty-entries"],
)
def test_terms_in_text_empty_input_returns_empty_tuple(
    text: str, entries: tuple[GlossaryEntry, ...]
) -> None:
    # Act
    found = terms_in_text(text, LanguageMap.ENGLISH, entries)

    # Assert
    assert found == ()


def test_terms_in_text_returns_each_entry_once_in_entries_order() -> None:
    # Arrange
    text = "The APC led. IFV and IFV, then an infantry fighting vehicle."

    # Act
    found = terms_in_text(text, LanguageMap.ENGLISH, (_IFV, _APC))

    # Assert
    assert found == (_IFV, _APC)


def test_simplify_entries_changes_only_the_zh_field() -> None:
    # Arrange
    entries = (
        make_entry("IFV", "infantry fighting vehicle", "步兵戰車"),
        make_entry(None, "course of action", "行動方案"),
    )
    converter = FakeScriptConverter({"戰": "战", "車": "车"})

    # Act
    simplified = simplify_entries(entries, converter)

    # Assert
    assert simplified == (
        make_entry("IFV", "infantry fighting vehicle", "步兵战车"),
        make_entry(None, "course of action", "行動方案"),
    )
