"""Pure functions that find reference-glossary entries in a text.

Pure logic: no I/O. English text is matched by abbreviation or full name,
Chinese text by the entry's expected translation (ADR-0012).
"""

import dataclasses
import functools
import re

from sentinel_rosetta.domain.interfaces import ScriptConverter
from sentinel_rosetta.domain.models import GlossaryEntry, LanguageMap

_MIN_FULL_NAME_WORDS = 2
_QUALIFIER_START = "（"


@functools.cache
def _whole_token_pattern(term: str, is_case_ignored: bool) -> "re.Pattern[str]":
    flags = re.IGNORECASE if is_case_ignored else 0
    return re.compile(
        r"(?<![\w-])" + re.escape(term) + r"(?:s|es)?(?![\w-])", flags
    )


def _expected_zh(entry: GlossaryEntry) -> str:
    return entry.zh.split(_QUALIFIER_START, 1)[0].strip()


def _appears_in_english(text: str, entry: GlossaryEntry) -> bool:
    if entry.abbreviation and _whole_token_pattern(
        entry.abbreviation, False
    ).search(text):
        return True
    return len(entry.full_name.split()) >= _MIN_FULL_NAME_WORDS and bool(
        _whole_token_pattern(entry.full_name, True).search(text)
    )


def _appears_in_mandarin(text: str, entry: GlossaryEntry) -> bool:
    expected_zh = _expected_zh(entry)
    return bool(expected_zh) and expected_zh in text


def terms_in_text(
    text: str, language: LanguageMap, entries: tuple[GlossaryEntry, ...]
) -> tuple[GlossaryEntry, ...]:
    if language == LanguageMap.ENGLISH:
        return tuple(
            entry for entry in entries if _appears_in_english(text, entry)
        )
    return tuple(
        entry for entry in entries if _appears_in_mandarin(text, entry)
    )


def simplify_entries(
    entries: tuple[GlossaryEntry, ...], script_converter: ScriptConverter
) -> tuple[GlossaryEntry, ...]:
    return tuple(
        dataclasses.replace(entry, zh=script_converter.to_simplified(entry.zh))
        for entry in entries
    )
