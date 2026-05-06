"""Seedable synthetic corpus generator. Reproducible, zero real PII.

Each example is a JSON object with:
  text:   str
  spans:  list[{label, start, end}]   # ground-truth char-offset spans

Labels match what the OpenAI Privacy Filter emits (`private_email`,
`private_phone`, `private_person`) plus regex-canonical labels (`EMAIL`,
`PHONE_NANP`, `PHONE_INTL`) so detectors that emit either form can score.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CLUE_RICH_TEMPLATES = [
    "Hi, my name is {person}. You can reach me at {email}.",
    "Please contact {person} at {phone}.",
    "Send the report to {email} by tomorrow.",
    "Call me back at {phone} when you get a chance.",
    "{person}'s phone is {phone}; their email is {email}.",
]

CLUE_LIGHT_TEMPLATES = [
    "{phone}",
    "{email}",
    "from: {email}",
    "{person}, {phone}",
    "{phone}\n{email}",
]

PERSONS = ["Alice Smith", "Bob Lee", "Carol Diaz", "Dan Brown", "Eve Park"]
EMAILS = [
    "alice@example.com", "bob.lee@corp.test", "carol_d@mail.invalid",
    "dan@brown.example", "eve.park@uni.test",
]
PHONES_NANP = ["555-867-5309", "(415) 555-1234", "212-555-9876", "800-555-0199"]
PHONES_INTL = ["+44 20 7946 0958", "+81 3-5555-1234", "+1-415-555-1234"]


def _label_for_phone(phone: str) -> str:
    return "PHONE_INTL" if phone.startswith("+") else "PHONE_NANP"


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    half = n // 2
    for i in range(n):
        templates = CLUE_RICH_TEMPLATES if i < half else CLUE_LIGHT_TEMPLATES
        tpl = rng.choice(templates)
        person = rng.choice(PERSONS)
        email = rng.choice(EMAILS)
        phone = rng.choice(PHONES_NANP + PHONES_INTL)
        text = tpl.format(person=person, email=email, phone=phone)
        spans = []
        if "{person}" in tpl:
            s = text.find(person)
            spans.append({"label": "private_person", "start": s, "end": s + len(person)})
        if "{email}" in tpl:
            s = text.find(email)
            spans.append({"label": "private_email", "start": s, "end": s + len(email)})
            spans.append({"label": "EMAIL", "start": s, "end": s + len(email)})
        if "{phone}" in tpl:
            s = text.find(phone)
            spans.append({"label": "private_phone", "start": s, "end": s + len(phone)})
            spans.append({"label": _label_for_phone(phone), "start": s, "end": s + len(phone)})
        out.append({"text": text, "spans": spans})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic eval corpus")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic.jsonl")
    args = parser.parse_args()
    examples = generate(args.n, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")
    print(f"wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
