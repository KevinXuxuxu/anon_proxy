"""Offline detector evaluation against a labeled JSONL corpus.

Run: `uv run python -m anon_proxy.eval [--corpus PATH] [--detectors LIST]`

Inputs are JSONL lines: {"text": "...", "spans": [{"label": str, "start": int, "end": int}, ...]}
Two corpora ship in `anon_proxy/eval_corpus/`: `synthetic.jsonl` (200 examples,
seedable generator) and `opf_samples.jsonl` (5 examples, openai/privacy-filter,
Apache 2.0 — smoke test only).

Labels from three vocabularies (ML `private_*`, regex `EMAIL`/`PHONE_NANP`/...,
OPF lowercase `email`/`phone`/...) are canonicalized to a common form before
matching. See `canonical_label`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from anon_proxy.privacy_filter import PrivacyFilter
from anon_proxy.telemetry import default_detector as default_baseline_detector


# Map any input label (ML / regex / OPF) to a canonical form for matching.
# Order is checked: exact match in CANONICAL_MAP first, then prefix rules.
CANONICAL_MAP: dict[str, str] = {
    "EMAIL": "EMAIL",
    "PHONE_NANP": "PHONE",
    "PHONE_INTL": "PHONE",
    "PHONE_LOOSE": "PHONE",
    "SSN": "SSN",
    "IPV4": "IP",
    "IPV6": "IP",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_person": "PERSON",
    "private_address": "ADDRESS",
    "private_account_number": "ACCOUNT",
    "private_date": "DATE",
    "private_url": "URL",
    "private_secret": "SECRET",
    "private_ssn": "SSN",
    "private_ip": "IP",
    "email": "EMAIL",
    "phone": "PHONE",
    "person": "PERSON",
    "address": "ADDRESS",
    "account_number": "ACCOUNT",
    "date": "DATE",
    "url": "URL",
    "secret": "SECRET",
    "ssn": "SSN",
    "ip": "IP",
}


def canonical_label(label: str) -> str:
    return CANONICAL_MAP.get(label, label)


def _dedupe_canonical(spans: list[Span]) -> list[Span]:
    """Drop spans that share a canonical key with an earlier span.

    Both bundled corpora carry per-span duplicates that canonicalize to the
    same label (synthetic emits both `private_email` and `EMAIL`; the default
    baseline regex emits both `PHONE_NANP` and `PHONE_LOOSE`). Counting them
    separately makes precision and recall wrong out of the box.
    """
    seen: set[tuple[str, int, int]] = set()
    out: list[Span] = []
    for s in spans:
        key = (canonical_label(s.label), s.start, s.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class LabeledExample:
    text: str
    truth: list[Span]
    predictions: list[Span]


def span_match(pred: Span, truth: Span) -> bool:
    if pred.start != truth.start or pred.end != truth.end:
        return False
    return canonical_label(pred.label) == canonical_label(truth.label)


def compute_metrics(examples: list[LabeledExample]) -> dict[str, dict[str, float]]:
    """Per-label precision / recall / F1 / n.

    n = number of ground-truth spans of that label. A label appears in the
    output if either truth or predictions contain it (after canonicalization).
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)

    for ex in examples:
        ex_truth = _dedupe_canonical(ex.truth)
        ex_predictions = _dedupe_canonical(ex.predictions)
        truth_remaining = list(ex_truth)
        for t in ex_truth:
            n[canonical_label(t.label)] += 1
        for p in ex_predictions:
            label_key = canonical_label(p.label)
            matched = next(
                (t for t in truth_remaining if span_match(p, t)),
                None,
            )
            if matched is not None:
                tp[canonical_label(matched.label)] += 1
                truth_remaining.remove(matched)
            else:
                fp[label_key] += 1
        for t in truth_remaining:
            fn[canonical_label(t.label)] += 1

    out: dict[str, dict[str, float]] = {}
    for label in set(tp) | set(fp) | set(fn) | set(n):
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": float(n[label]),
        }
    return out


def load_corpus(path: Path) -> list[LabeledExample]:
    out: list[LabeledExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            truth = [Span(label=s["label"], start=s["start"], end=s["end"]) for s in rec["spans"]]
            out.append(LabeledExample(text=rec["text"], truth=truth, predictions=[]))
    return out


def run_detector(name: str, examples: list[LabeledExample]) -> list[LabeledExample]:
    if name == "ml":
        det = PrivacyFilter()
        predict = lambda text: [
            Span(label=e.label, start=e.start, end=e.end) for e in det.detect(text)
        ]
    elif name == "baseline":
        det = default_baseline_detector()
        predict = lambda text: [
            Span(label=e.label, start=e.start, end=e.end) for e in det.detect(text)
        ]
    else:
        raise ValueError(f"unknown detector {name!r} (expected: ml, baseline)")
    return [
        LabeledExample(text=ex.text, truth=ex.truth, predictions=predict(ex.text))
        for ex in examples
    ]


def format_report(detector_name: str, metrics: dict[str, dict[str, float]]) -> str:
    lines = [f"{detector_name}:"]
    for label in sorted(metrics):
        m = metrics[label]
        lines.append(
            f"  {label:14s}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
            f"F1={m['f1']:.2f}  (n={int(m['n'])})"
        )
    return "\n".join(lines)


def main() -> None:
    default_corpus = Path(__file__).parent / "eval_corpus" / "synthetic.jsonl"
    parser = argparse.ArgumentParser(
        description="Offline detector eval against a labeled JSONL corpus.",
    )
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument(
        "--detectors",
        type=str,
        default="ml,baseline",
        help="Comma-separated subset of {ml, baseline}",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    examples = load_corpus(args.corpus)
    print(f"Corpus: {args.corpus}  ({len(examples)} examples)\n")

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name in [d.strip() for d in args.detectors.split(",") if d.strip()]:
        scored = run_detector(name, examples)
        m = compute_metrics(scored)
        all_metrics[name] = m
        print(format_report(name, m))
        print()

    if args.report:
        args.report.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
