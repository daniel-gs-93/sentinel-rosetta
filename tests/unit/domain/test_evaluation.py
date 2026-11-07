import pytest

from sentinel_rosetta.domain.evaluation import (
    build_evaluation_report,
    chrf_score,
)
from sentinel_rosetta.domain.models import LanguageMap, TrainingPair
from tests.doubles import make_entry

# Reference values from sacrebleu 2.6.0:
# CHRF().corpus_score([hypothesis], [[reference]]).score
_SACREBLEU_CASES = [
    (
        "解放军使用多种装甲运兵车。",
        "解放军使用各种步兵战车。",
        30.293876269974064,
    ),
    (
        "The UAV returned to base.",
        "The drone returned to the base.",
        47.65496051691024,
    ),
    ("abc", "abd", 38.888888888888886),
    (
        "集团军是基本作战编制",
        "集团军是解放军陆军的基本作战层级编制。",
        23.633146747595802,
    ),
]

# Glossary entries already converted to Simplified Chinese, as the use case
# hands them to the report.
_IFV = make_entry("IFV", "infantry fighting vehicle", "步兵战车")
_APC = make_entry("APC", "armored personnel carrier", "装甲运兵车")
_US = make_entry("U.S.", "United States", "美国")


def test_chrf_score_identical_texts_returns_100() -> None:
    # Act
    score = chrf_score(("步兵战车已经到达基地。",), ("步兵战车已经到达基地。",))

    # Assert
    assert score == pytest.approx(100.0)


def test_chrf_score_no_shared_characters_returns_zero() -> None:
    # Act
    score = chrf_score(("abc",), ("xyz",))

    # Assert
    assert score == 0.0


@pytest.mark.parametrize(
    ("hypothesis", "reference", "expected_score"),
    _SACREBLEU_CASES,
    ids=["chinese", "english", "short-text", "different-lengths"],
)
def test_chrf_score_matches_sacrebleu_reference_values(
    hypothesis: str, reference: str, expected_score: float
) -> None:
    # Act
    score = chrf_score((hypothesis,), (reference,))

    # Assert
    assert score == pytest.approx(expected_score, abs=1e-6)


def test_chrf_score_ignores_whitespace_differences() -> None:
    # Act
    score = chrf_score(
        ("The UAV  returned\nto base.",), ("The UAV returned to base.",)
    )

    # Assert
    assert score == pytest.approx(100.0)


def test_chrf_score_sums_statistics_across_sentences() -> None:
    # Arrange: sacrebleu 2.6.0 corpus_score of the first and last cases.
    hypotheses = (_SACREBLEU_CASES[0][0], _SACREBLEU_CASES[3][0])
    references = (_SACREBLEU_CASES[0][1], _SACREBLEU_CASES[3][1])
    mean_of_sentence_scores = (
        _SACREBLEU_CASES[0][2] + _SACREBLEU_CASES[3][2]
    ) / 2

    # Act
    score = chrf_score(hypotheses, references)

    # Assert
    assert score == pytest.approx(26.295025931497722, abs=1e-6)
    assert score != pytest.approx(mean_of_sentence_scores, abs=1e-3)


def test_chrf_score_empty_hypothesis_returns_zero() -> None:
    # Act
    score = chrf_score(("",), ("abc",))

    # Assert
    assert score == 0.0


def test_chrf_score_length_mismatch_raises_value_error() -> None:
    # Act / Assert
    with pytest.raises(ValueError):
        chrf_score(("a", "b"), ("a",))


def test_build_report_scores_follow_chrf_score_function() -> None:
    # Arrange
    pairs = (
        TrainingPair("source one", "解放军使用各种步兵战车。"),
        TrainingPair("source two", "集团军是解放军陆军的基本作战层级编制。"),
    )
    base_outputs = ("解放军使用多种装甲运兵车。", "集团军是基本作战编制")
    tuned_outputs = (
        "解放军使用各种步兵战车。",
        "集团军是解放军陆军的基本作战层级编制。",
    )

    # Act
    report = build_evaluation_report(
        pairs,
        base_outputs,
        tuned_outputs,
        (),
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
    )

    # Assert
    references = tuple(pair.target_text for pair in pairs)
    assert report.sample_count == 2
    assert report.base_score == pytest.approx(
        chrf_score(base_outputs, references)
    )
    assert report.tuned_score == pytest.approx(100.0)
    assert [row.source_text for row in report.rows] == [
        "source one",
        "source two",
    ]
    assert report.rows[0].base_score == pytest.approx(
        30.293876269974064, abs=1e-6
    )
    assert report.rows[0].tuned_output == tuned_outputs[0]
    assert report.rows[0].reference_text == references[0]


def test_build_report_length_mismatch_raises_value_error() -> None:
    # Act / Assert
    with pytest.raises(ValueError):
        build_evaluation_report(
            (TrainingPair("a", "b"),),
            ("x", "y"),
            ("z",),
            (),
            LanguageMap.ENGLISH,
            LanguageMap.MANDARIN,
        )


def test_build_report_en_to_zh_counts_terms_per_row() -> None:
    # Arrange
    pairs = (TrainingPair("The IFV arrived.", "步兵战车到达。"),)

    # Act
    report = build_evaluation_report(
        pairs,
        ("战车到达。",),
        ("步兵战车到达。",),
        (_IFV,),
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
    )

    # Assert
    row = report.rows[0]
    assert (
        row.expected_term_count,
        row.base_hit_count,
        row.tuned_hit_count,
    ) == (1, 0, 1)


def test_build_report_zh_to_en_counts_terms_found_in_english_outputs() -> None:
    # Arrange
    pairs = (TrainingPair("步兵战车到达。", "The IFV arrived."),)

    # Act
    report = build_evaluation_report(
        pairs,
        ("The vehicle arrived.",),
        ("The IFV arrived.",),
        (_IFV,),
        LanguageMap.MANDARIN,
        LanguageMap.ENGLISH,
    )

    # Assert
    row = report.rows[0]
    assert (
        row.expected_term_count,
        row.base_hit_count,
        row.tuned_hit_count,
    ) == (1, 0, 1)


def test_build_report_term_missing_from_reference_is_not_expected() -> None:
    # Arrange: the teacher translated "U.S." as 美军, not the glossary's 美国.
    pairs = (TrainingPair("The U.S. Army arrived.", "美军到达。"),)

    # Act
    report = build_evaluation_report(
        pairs,
        ("美军到达。",),
        ("美国军队到达。",),
        (_US,),
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
    )

    # Assert
    row = report.rows[0]
    assert (
        row.expected_term_count,
        row.base_hit_count,
        row.tuned_hit_count,
    ) == (0, 0, 0)


def test_build_report_term_counts_sum_over_rows() -> None:
    # Arrange: row counts (expected, base, tuned) are (1, 0, 1), (2, 1, 2)
    # and (0, 0, 0).
    pairs = (
        TrainingPair("The IFV arrived.", "步兵战车到达。"),
        TrainingPair("The IFV and APC arrived.", "步兵战车和装甲运兵车到达。"),
        TrainingPair("Nothing here.", "这里什么都没有。"),
    )
    base_outputs = ("到达。", "步兵战车到达。", "这里没有。")
    tuned_outputs = (
        "步兵战车到达。",
        "步兵战车和装甲运兵车到达。",
        "这里没有。",
    )

    # Act
    report = build_evaluation_report(
        pairs,
        base_outputs,
        tuned_outputs,
        (_IFV, _APC),
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
    )

    # Assert
    assert (
        report.expected_term_count,
        report.base_hit_count,
        report.tuned_hit_count,
    ) == (3, 1, 3)


def test_build_report_without_entries_has_zero_term_counts() -> None:
    # Arrange
    pairs = (TrainingPair("The IFV arrived.", "步兵战车到达。"),)

    # Act
    report = build_evaluation_report(
        pairs,
        ("到达。",),
        ("步兵战车到达。",),
        (),
        LanguageMap.ENGLISH,
        LanguageMap.MANDARIN,
    )

    # Assert
    assert (
        report.expected_term_count,
        report.base_hit_count,
        report.tuned_hit_count,
    ) == (0, 0, 0)
    assert report.rows[0].expected_term_count == 0
