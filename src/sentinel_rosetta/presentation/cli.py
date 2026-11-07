"""Resource-aware: command-line entry point that reads argv, writes stdout."""

import argparse
import sys
from typing import Optional

from sentinel_rosetta.presentation.bootstrap import (
    build_translate_text_use_case,
)
from sentinel_rosetta.presentation.dtos import TranslationRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("text")
    translate_parser.add_argument("-s", "--source-lang", required=True)
    translate_parser.add_argument("-t", "--target-lang", required=True)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        request = TranslationRequest(
            text=args.text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
        use_case = build_translate_text_use_case()
        result = use_case.execute(
            request.text, request.source_lang, request.target_lang
        )
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    print(result.target_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
