"""Resource-aware: command-line entry point for record validation."""

import argparse
import sys
from typing import Optional

from sentinel_rosetta.presentation.validate_generated_records_bootstrap import (
    build_validate_generated_records_use_case,
)


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=(
            "python -m sentinel_rosetta.presentation."
            "validate_generated_records_cli"
        ),
        description=(
            "Check the field format of every JSONL file in each direction "
            "folder of data/llm_generated/."
        ),
    )


def main(argv: Optional[list[str]] = None) -> int:
    _build_parser().parse_args(argv)

    try:
        record_count = build_validate_generated_records_use_case().execute()
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"checked {record_count} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
