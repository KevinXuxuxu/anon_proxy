"""Chat with Claude through the PII mask/unmask layer.

Requires ANTHROPIC_API_KEY in the environment. Each turn:
  1. Your message is masked (PII -> placeholder tokens) before being sent.
  2. The full conversation history sent to Claude stays masked throughout.
  3. Claude's streamed reply is printed live (masked, dim), then the
     rendered version with placeholders substituted back to originals.

Input is multi-line: plain Enter inserts a newline, Alt+Enter submits.
Ctrl-D exits. See _make_prompt_session() for how to remap VS Code's
Shift+Enter onto Alt+Enter so it also submits.

With --no-mask, the local masker is skipped entirely — useful for pointing
this script at anon-proxy (ANTHROPIC_BASE_URL=http://127.0.0.1:8080) to
exercise the server-side masking instead.

Usage:
  uv run python test_mask.py
  uv run python test_mask.py --show-store
  uv run python test_mask.py --model claude-sonnet-4-6
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 uv run python test_mask.py --no-mask
"""

from __future__ import annotations

import argparse
import os
import sys

import anthropic
from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.key_binding import KeyBindings

from anon_proxy import Masker, PrivacyFilter, RegexDetector, load_merge_gap, load_patterns

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user's messages may contain placeholder "
    "tokens like <PERSON_1>, <EMAIL_1>, <PHONE_1>, <ADDRESS_1>, <DATE_1>, "
    "<ACCOUNT_NUMBER_1>, etc. Each token is an opaque reference to a real "
    "private value that has been redacted. Two occurrences of the same token "
    "always refer to the same entity. When you need to refer to one of these "
    "entities in your reply, use the token verbatim - do NOT invent real "
    "names, emails, phone numbers, or other values, and do NOT rewrite tokens "
    "as generic labels like [REDACTED]. The user will see the original values "
    "re-inserted into your response."
)

DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def dump_store(masker: Masker) -> None:
    items = masker.store.items()
    if not items:
        print(f"{DIM}  (store empty){RESET}")
        return
    print(f"{DIM}  store:{RESET}")
    for token, original in items:
        print(f"    {token}  ->  {original!r}")


def _make_prompt_session() -> PromptSession:
    """Multi-line input: plain Enter inserts a newline; Alt+Enter submits.

    Most terminals don't send a distinct code for Shift+Enter — it emits the
    same byte as plain Enter, so we can't bind it directly. To get Shift+Enter
    to submit in VS Code's integrated terminal, add this to keybindings.json:

        {
          "key": "shift+enter",
          "command": "workbench.action.terminal.sendSequence",
          "args": { "text": "\\u001b\\r" },
          "when": "terminalFocus"
        }

    That remaps Shift+Enter to Alt+Enter, which this prompt already accepts.
    """
    return PromptSession(multiline=True, key_bindings=KeyBindings())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Claude model ID (default: claude-opus-4-7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Per-response token ceiling (default: 8192).",
    )
    parser.add_argument(
        "--show-store",
        action="store_true",
        help="Dump the PII mapping after each turn (ignored with --no-mask).",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip local masking/unmasking; send raw text and display raw replies. "
             "Pair with ANTHROPIC_BASE_URL to test the proxy's own masking.",
    )
    parser.add_argument(
        "--patterns",
        default=None,
        help="Path to a JSON file of additional regex patterns (label -> regex).",
    )
    parser.add_argument(
        "--merge-gap-file",
        default=None,
        help="Path to a JSON file of per-label merge-gap chars (label -> chars). "
             "Overrides entries in DEFAULT_MERGE_GAP_ALLOWED.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        metavar="N",
        help="Max characters per chunk fed to the model (default: 1500).",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            f"{RED}error:{RESET} ANTHROPIC_API_KEY is not set.\n"
            "Export your key and try again:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        return 2

    masker: Masker | None
    if args.no_mask:
        masker = None
    else:
        print("Loading openai/privacy-filter ...", file=sys.stderr)
        extra_detectors = []
        if args.patterns:
            try:
                patterns = load_patterns(args.patterns)
            except (OSError, ValueError) as e:
                print(f"{RED}error:{RESET} {e}", file=sys.stderr)
                return 2
            extra_detectors.append(RegexDetector(patterns))
        pf: PrivacyFilter | None = None
        if args.merge_gap_file or args.chunk_size != 1500:
            merge_gap = None
            if args.merge_gap_file:
                try:
                    merge_gap = load_merge_gap(args.merge_gap_file)
                except (OSError, ValueError) as e:
                    print(f"{RED}error:{RESET} {e}", file=sys.stderr)
                    return 2
            pf = PrivacyFilter(merge_gap_allowed=merge_gap, chunk_size=args.chunk_size)
        masker = Masker(filter=pf, extra_detectors=extra_detectors)
    client = anthropic.Anthropic()
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    status_bits = [f"model={args.model}"]
    status_bits.append("masking=off" if args.no_mask else "masking=local")
    if base_url:
        status_bits.append(f"base_url={base_url}")
    print(
        f"Ready. {' | '.join(status_bits)}.\n"
        f"  Enter = newline.  Alt+Enter = submit.  Ctrl-D or empty submit = exit.\n",
        file=sys.stderr,
    )
    session = _make_prompt_session()

    def do_mask(text: str) -> str:
        return masker.mask(text) if masker is not None else text

    def do_unmask(text: str) -> str:
        return masker.unmask(text) if masker is not None else text

    history: list[dict] = []
    turn = 1
    while True:
        try:
            user_text = session.prompt(ANSI(f"{CYAN}you[{turn}]>{RESET} "))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text.strip():
            return 0

        sent_text = do_mask(user_text)
        if sent_text != user_text:
            print(f"  {DIM}sending -> {sent_text}{RESET}")

        history.append({"role": "user", "content": sent_text})

        try:
            with client.messages.stream(
                model=args.model,
                max_tokens=args.max_tokens,
                system=SYSTEM_PROMPT,
                messages=history,
                cache_control={"type": "ephemeral"},
            ) as stream:
                print(f"{CYAN}claude[{turn}]>{RESET} {DIM}", end="", flush=True)
                for chunk in stream.text_stream:
                    print(chunk, end="", flush=True)
                print(RESET)
                final = stream.get_final_message()
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}interrupted{RESET}\n")
            history.pop()
            continue
        except anthropic.APIError as e:
            history.pop()
            print(f"  {RED}API error:{RESET} {e}\n")
            continue

        assistant_text = "".join(b.text for b in final.content if b.type == "text")
        history.append({"role": "assistant", "content": final.content})

        rendered = do_unmask(assistant_text)
        if rendered != assistant_text:
            print(f"  {DIM}rendered ->{RESET} {rendered}")

        usage = final.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        print(
            f"  {DIM}usage: in={usage.input_tokens} out={usage.output_tokens}"
            f" cache_read={cache_read} cache_write={cache_write}{RESET}"
        )

        if args.show_store and masker is not None:
            dump_store(masker)

        print()
        turn += 1


if __name__ == "__main__":
    raise SystemExit(main())
