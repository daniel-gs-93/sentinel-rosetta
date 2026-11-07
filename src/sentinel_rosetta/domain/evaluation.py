"""Pure logic: chrF scoring and term-hit counting for the hold-out report."""

from collections import Counter

from sentinel_rosetta.domain.glossary_terms import terms_in_text
from sentinel_rosetta.domain.models import (
    EvaluationReport,
    EvaluationRow,
    GlossaryEntry,
    LanguageMap,
    TrainingPair,
)

_CHAR_ORDER = 6
_BETA_SQUARED = 4.0


def _character_ngram_counts(text: str, order: int) -> Counter[str]:
    squeezed_text = "".join(text.split())
    return Counter(
        squeezed_text[start : start + order]
        for start in range(len(squeezed_text) - order + 1)
    )


def chrf_score(
    hypotheses: tuple[str, ...], references: tuple[str, ...]
) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")

    hypothesis_totals = [0] * _CHAR_ORDER
    reference_totals = [0] * _CHAR_ORDER
    match_totals = [0] * _CHAR_ORDER
    for hypothesis, reference in zip(hypotheses, references):
        for order_index in range(_CHAR_ORDER):
            order = order_index + 1
            hypothesis_counts = _character_ngram_counts(hypothesis, order)
            reference_counts = _character_ngram_counts(reference, order)
            hypothesis_totals[order_index] += sum(hypothesis_counts.values())
            reference_totals[order_index] += sum(reference_counts.values())
            match_totals[order_index] += sum(
                (hypothesis_counts & reference_counts).values()
            )

    precision_sum = 0.0
    recall_sum = 0.0
    effective_order = 0
    for hypothesis_total, reference_total, match_total in zip(
        hypothesis_totals, reference_totals, match_totals
    ):
        if hypothesis_total > 0 and reference_total > 0:
            precision_sum += match_total / hypothesis_total
            recall_sum += match_total / reference_total
            effective_order += 1
    if effective_order == 0:
        return 0.0

    average_precision = precision_sum / effective_order
    average_recall = recall_sum / effective_order
    if average_precision + average_recall == 0:
        return 0.0
    return (
        100
        * (1 + _BETA_SQUARED)
        * average_precision
        * average_recall
        / (_BETA_SQUARED * average_precision + average_recall)
    )


def _count_hits(
    expected_entries: tuple[GlossaryEntry, ...],
    output: str,
    target_lang: LanguageMap,
) -> int:
    return len(terms_in_text(output, target_lang, expected_entries))


def _build_row(
    pair: TrainingPair,
    base_output: str,
    tuned_output: str,
    entries: tuple[GlossaryEntry, ...],
    source_lang: LanguageMap,
    target_lang: LanguageMap,
) -> EvaluationRow:
    source_entries = terms_in_text(pair.source_text, source_lang, entries)
    reference_entries = terms_in_text(pair.target_text, target_lang, entries)
    expected_entries = tuple(
        entry for entry in source_entries if entry in reference_entries
    )
    return EvaluationRow(
        source_text=pair.source_text,
        reference_text=pair.target_text,
        base_output=base_output,
        tuned_output=tuned_output,
        base_score=chrf_score((base_output,), (pair.target_text,)),
        tuned_score=chrf_score((tuned_output,), (pair.target_text,)),
        expected_term_count=len(expected_entries),
        base_hit_count=_count_hits(expected_entries, base_output, target_lang),
        tuned_hit_count=_count_hits(
            expected_entries, tuned_output, target_lang
        ),
    )


def build_evaluation_report(
    holdout_pairs: tuple[TrainingPair, ...],
    base_outputs: tuple[str, ...],
    tuned_outputs: tuple[str, ...],
    entries: tuple[GlossaryEntry, ...],
    source_lang: LanguageMap,
    target_lang: LanguageMap,
) -> EvaluationReport:
    if not len(holdout_pairs) == len(base_outputs) == len(tuned_outputs):
        raise ValueError("pairs and outputs must have the same length")

    references = tuple(pair.target_text for pair in holdout_pairs)
    rows = tuple(
        _build_row(
            pair, base_output, tuned_output, entries, source_lang, target_lang
        )
        for pair, base_output, tuned_output in zip(
            holdout_pairs, base_outputs, tuned_outputs
        )
    )
    return EvaluationReport(
        sample_count=len(rows),
        base_score=chrf_score(base_outputs, references),
        tuned_score=chrf_score(tuned_outputs, references),
        expected_term_count=sum(row.expected_term_count for row in rows),
        base_hit_count=sum(row.base_hit_count for row in rows),
        tuned_hit_count=sum(row.tuned_hit_count for row in rows),
        rows=rows,
    )
