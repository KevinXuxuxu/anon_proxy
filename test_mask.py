"""Chat with Claude through the PII mask/unmask layer.

Requires ANTHROPIC_API_KEY in the environment. Each turn:
  1. Your message is masked (PII -> placeholder tokens) before being sent.
  2. The full conversation history sent to Claude stays masked throughout.
  3. Claude's streamed reply is printed live (masked, dim), then the
     rendered version with placeholders substituted back to originals.

Usage:
  uv run python test_mask.py
  uv run python test_mask.py --show-store
  uv run python test_mask.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import os
import sys

import anthropic

from anon_proxy import Masker

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
        help="Dump the PII mapping after each turn.",
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

    print("Loading openai/privacy-filter ...", file=sys.stderr)
    masker = Masker()
    client = anthropic.Anthropic()
    print(
        f"Ready. Model: {args.model}. Blank line or Ctrl-D to exit.\n",
        file=sys.stderr,
    )

    history: list[dict] = []
    turn = 1
    while True:
        try:
            user_text = input(f"{CYAN}you[{turn}]>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text.strip():
            return 0

        masked_user = masker.mask(user_text)
        if masked_user != user_text:
            print(f"  {DIM}sending -> {masked_user}{RESET}")

        history.append({"role": "user", "content": masked_user})

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

        assistant_masked = "".join(b.text for b in final.content if b.type == "text")
        history.append({"role": "assistant", "content": final.content})

        unmasked = masker.unmask(assistant_masked)
        if unmasked != assistant_masked:
            print(f"  {DIM}rendered ->{RESET} {unmasked}")

        usage = final.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        print(
            f"  {DIM}usage: in={usage.input_tokens} out={usage.output_tokens}"
            f" cache_read={cache_read} cache_write={cache_write}{RESET}"
        )

        if args.show_store:
            dump_store(masker)

        print()
        turn += 1


if __name__ == "__main__":
    raise SystemExit(main())
