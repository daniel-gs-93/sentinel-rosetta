"""Resource-aware: command-line entry point for fine-tuning."""

import argparse
import sys
from typing import Optional

from sentinel_rosetta.domain.models import EvaluationReport
from sentinel_rosetta.presentation.dtos import FinetuneRequest
from sentinel_rosetta.presentation.finetune_bootstrap import (
    build_finetune_model_use_case,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_rosetta.presentation.finetune_cli",
        description=(
            "Fine-tune one Opus-MT translation direction on the matching "
            "folder of data/llm_generated/."
        ),
    )
    parser.add_argument("-s", "--source-lang", required=True)
    parser.add_argument("-t", "--target-lang", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--holdout-fraction", type=float)
    parser.add_argument("--base-model-dir")
    parser.add_argument("--output-dir")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    arguments = {
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "holdout_fraction": args.holdout_fraction,
        "base_model_dir": args.base_model_dir,
        "output_dir": args.output_dir,
    }
    provided_arguments = {
        name: value for name, value in arguments.items() if value is not None
    }

    try:
        request = FinetuneRequest.model_validate(provided_arguments)
        use_case = build_finetune_model_use_case(
            request.source_lang,
            request.target_lang,
            request.resolved_base_model_dir(),
            request.resolved_output_dir(),
            request.max_length,
        )
        result = use_case.execute(
            request.source_lang,
            request.target_lang,
            request.to_training_config(),
            request.holdout_fraction,
        )
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    print(
        f"fine-tuned {result.pair_count} pairs, "
        f"train_loss={result.train_loss:.4f}, "
        f"saved to {result.output_dir}"
    )
    if result.evaluation is not None:
        _print_evaluation(result.evaluation)
    return 0


def _print_evaluation(report: EvaluationReport) -> None:
    print(
        f"held-out {report.sample_count} pairs: "
        f"chrF base={report.base_score:.2f}, tuned={report.tuned_score:.2f}"
    )
    print(
        f"terms: base={report.base_hit_count}/{report.expected_term_count}, "
        f"tuned={report.tuned_hit_count}/{report.expected_term_count}"
    )
    for row_number, row in enumerate(report.rows, start=1):
        print(
            f"[{row_number}] chrF base={row.base_score:.2f} "
            f"tuned={row.tuned_score:.2f}",
            file=sys.stderr,
        )
        print(f"  source   : {row.source_text}", file=sys.stderr)
        print(f"  reference: {row.reference_text}", file=sys.stderr)
        print(f"  base     : {row.base_output}", file=sys.stderr)
        print(f"  tuned    : {row.tuned_output}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
