"""Interactive tester for the openai/privacy-filter model.

Examples:
    uv run python test_filter.py "My name is Alice Smith"
    echo "Email me at bob@acme.com" | uv run python test_filter.py -
    uv run python test_filter.py            # REPL mode
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from anon_proxy.privacy_filter import PIIEntity, PrivacyFilter

YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def highlight(text: str, entities: Sequence[PIIEntity]) -> str:
    if not entities:
        return text
    out = text
    for e in sorted(entities, key=lambda x: x.start, reverse=True):
        tag = f"{YELLOW}[{e.label}:{e.text}]{RESET}"
        out = out[: e.start] + tag + out[e.end :]
    return out


def print_analysis(text: str, entities: Sequence[PIIEntity]) -> None:
    print(highlight(text, entities))
    if not entities:
        print(f"{DIM}  (no PII detected){RESET}")
        return
    print()
    for e in entities:
        print(f"  {e.label:<12} {e.text!r:<30} score={e.score:.3f}  offset={e.start}-{e.end}")


def _json_default(o):
    # Pipeline scores come back as numpy scalars; unwrap to plain Python.
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Not JSON-serializable: {type(o).__name__}")


def print_raw(raw: list[dict]) -> None:
    print(f"\n{DIM}raw pipeline output:{RESET}")
    print(json.dumps(raw, indent=2, default=_json_default, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to analyze. Omit for REPL, or pass '-' to read stdin.",
    )
    parser.add_argument(
        "--aggregation",
        default="simple",
        choices=["none", "simple", "first", "average", "max"],
        help="Pipeline aggregation strategy (default: simple).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging of adjacent same-label spans.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print the pipeline's raw JSON output.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cpu', 'cuda', 'cuda:0'. Default: auto.",
    )
    args = parser.parse_args()

    print(f"Loading {PrivacyFilter.MODEL_ID} ...", file=sys.stderr)
    pf = PrivacyFilter(
        aggregation_strategy=args.aggregation,
        merge_adjacent=not args.no_merge,
        device=args.device,
    )
    print("Ready.\n", file=sys.stderr)

    def analyze(text: str) -> None:
        print_analysis(text, pf.detect(text))
        if args.raw:
            print_raw(pf.detect_raw(text))

    if args.text == ["-"]:
        analyze(sys.stdin.read())
        return 0

    if args.text:
        analyze(" ".join(args.text))
        return 0

    print("Enter text to analyze. Blank line or Ctrl-D to exit.\n")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            return 0
        analyze(line)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
