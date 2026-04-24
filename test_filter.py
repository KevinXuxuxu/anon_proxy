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

from anon_proxy.privacy_filter import PIIEntity, PrivacyFilter, load_merge_gap

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


def _parse_merge_gap(items: list[str]) -> dict[str, str]:
    """Parse --merge-gap LABEL=CHARS occurrences into {LABEL: CHARS}.
    Last flag for a given label wins; put every allowed char in one flag.
    """
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--merge-gap expects LABEL=CHARS, got {item!r}")
        label, chars = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"--merge-gap has empty label in {item!r}")
        out[label] = chars
    return out


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
        "--merge-gap",
        action="append",
        default=[],
        metavar="LABEL=CHARS",
        help="Override the per-label merge-gap policy for one label. Repeatable. "
             "Whitespace is NOT implicit — include it in CHARS if you want it. "
             "Examples: --merge-gap PERSON=\" -'.\" --merge-gap ADDRESS=,.#",
    )
    parser.add_argument(
        "--merge-gap-file",
        default=None,
        metavar="PATH",
        help="Path to a JSON file of per-label merge-gap chars (label -> chars). "
             "Loaded first; individual --merge-gap flags override matching labels.",
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

    merge_gap_allowed: dict[str, str] = {}
    if args.merge_gap_file:
        try:
            merge_gap_allowed.update(load_merge_gap(args.merge_gap_file))
        except (OSError, ValueError) as e:
            parser.error(str(e))
    try:
        merge_gap_allowed.update(_parse_merge_gap(args.merge_gap))
    except ValueError as e:
        parser.error(str(e))

    print(f"Loading {PrivacyFilter.MODEL_ID} ...", file=sys.stderr)
    pf = PrivacyFilter(
        aggregation_strategy=args.aggregation,
        merge_adjacent=not args.no_merge,
        merge_gap_allowed=merge_gap_allowed,
        device=args.device,
    )
    if merge_gap_allowed:
        shown = ", ".join(f"{k}={v!r}" for k, v in merge_gap_allowed.items())
        print(f"  merge_gap_allowed: {shown}", file=sys.stderr)
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
