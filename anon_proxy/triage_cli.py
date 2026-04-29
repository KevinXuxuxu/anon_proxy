"""anon-proxy telemetry <subcmd> entry point.

Subcommand surface for inspecting and curating local PII telemetry. Designed
so that the LLM-driven triage skill can call zero-PII subsets safely — see
`triage --json --metadata-only` which strips encrypted fields entirely.

Subcommand overview:
  metrics        — zero-PII daily rollup stats; no key needed
  triage         — interactive or JSON review of raw records
                   --metadata-only: strips encrypted fields (skill-safe, no key)
                   --json:          decrypts + computes signatures, strips raw text
  label          — promote a raw record into the labeled corpus (8B)
  corpus         — inspect / export labeled corpus (8B)
  purge          — manually purge raw or corpus records (8B)
  export-key     — print base64 keyring key with loud warning (8B)
  import-key     — read base64 key from stdin into keyring (8B)
  suggest-regex  — mechanical regex synthesis from labeled corpus (8B)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from anon_proxy.storage_paths import default_data_dir


def _data_dir() -> Path:
    """Honor ANON_PROXY_DATA_DIR override; otherwise OS default."""
    return Path(os.environ.get("ANON_PROXY_DATA_DIR", str(default_data_dir())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="anon-proxy telemetry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_triage = sub.add_parser("triage", help="Interactively review interesting raw records")
    p_triage.add_argument("--days", type=int, default=7)
    p_triage.add_argument(
        "--source",
        choices=["baseline", "ml", "disagreement", "leak-back"],
        default=None,
    )
    p_triage.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable output (no interactive prompts)",
    )
    p_triage.add_argument(
        "--metadata-only",
        action="store_true",
        help="Omit encrypted text from output (skill-safe, no key required)",
    )

    p_label = sub.add_parser("label", help="Promote a raw record into the labeled corpus")
    p_label.add_argument("record_id")
    p_label.add_argument("--label", required=True)
    p_label.add_argument("--decision", choices=["keep", "drop"], required=True)

    p_corpus = sub.add_parser("corpus", help="Inspect or export the labeled corpus")
    p_corpus.add_argument("action", choices=["list", "show", "export"])
    p_corpus.add_argument("--id", default=None)

    p_purge = sub.add_parser("purge", help="Manually purge raw or corpus records")
    p_purge.add_argument("scope", choices=["raw", "corpus"])
    p_purge.add_argument("--before", default=None)
    p_purge.add_argument("--id", default=None)

    p_metrics = sub.add_parser("metrics", help="Show daily aggregate rollups (zero-PII)")
    p_metrics.add_argument("--since", default=None)
    p_metrics.add_argument("--json", action="store_true")

    sub.add_parser("export-key", help="Print base64 keyring key (loud warning)")
    sub.add_parser("import-key", help="Read base64 key from stdin and store in keyring")

    p_sg = sub.add_parser("suggest-regex", help="Mechanical regex synthesis from labeled corpus")
    p_sg.add_argument("--from-corpus", action="store_true")

    args = parser.parse_args(argv)
    handler = _HANDLERS[args.cmd]
    handler(args)


# ---------------------------------------------------------------------------
# Stub handler used for Phase 8B subcommands
# ---------------------------------------------------------------------------


def _stub(args):
    print(f"[stub] {args.cmd} args={vars(args)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 8.2: metrics handler
# ---------------------------------------------------------------------------


def _metrics_handler(args):
    metrics_path = _data_dir() / "metrics.jsonl"
    if not metrics_path.exists():
        print("No metrics yet.", file=sys.stderr)
        return
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if args.since:
        lines = [line for line in lines if line["date"] >= args.since]
    if args.json:
        print(json.dumps(lines, indent=2))
    else:
        for line in lines:
            print(
                f"{line['date']}: {line['total_records']} records, "
                f"leak_back={line.get('leak_back', 0)}, "
                f"labels={line.get('label_counts', {})}"
            )


# ---------------------------------------------------------------------------
# 8.3: triage handler (JSON paths)
# ---------------------------------------------------------------------------


def _matches_source(rec: dict, source: str) -> bool:
    """Return True if any span in rec matches the requested source filter."""
    if source == "leak-back":
        return any(s.get("side") == "response" for s in rec.get("spans", []))
    if source == "disagreement":
        # Has both ml and (baseline OR user_regex) span sources
        sources = {s.get("source") for s in rec.get("spans", [])}
        return "ml" in sources and bool(sources & {"baseline", "user_regex"})
    return any(s.get("source") == source for s in rec.get("spans", []))


def _triage_handler(args):
    raw_path = _data_dir() / "telemetry-raw.jsonl"
    if not raw_path.exists():
        print("No raw telemetry.", file=sys.stderr)
        return

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=args.days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    records = []
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["ts"] < cutoff:
            continue
        if args.source and not _matches_source(rec, args.source):
            continue
        records.append(rec)

    if args.json and args.metadata_only:
        # Strip every encrypted field — no key required (skill-safe path)
        for rec in records:
            rec.pop("enc_text", None)
            for span in rec.get("spans", []):
                span.pop("enc_text", None)
                span.pop("enc_window", None)
        print(json.dumps(records, indent=2))
        return

    if args.json:
        # Decrypt + compute structural signatures, then strip raw text
        from anon_proxy.crypto import resolve_key, decrypt_field
        from anon_proxy.signatures import compute_signature
        key = resolve_key()
        for rec in records:
            for span in rec.get("spans", []):
                if "enc_text" in span:
                    raw = decrypt_field(span["enc_text"], key)
                    span["signature"] = compute_signature(span["label"], raw)
                    span.pop("enc_text", None)
                    span.pop("enc_window", None)
            rec.pop("enc_text", None)
        print(json.dumps(records, indent=2))
        return

    # Interactive path (8.4)
    _interactive_triage(records, _data_dir())


# ---------------------------------------------------------------------------
# 8.4: Interactive triage UI
# ---------------------------------------------------------------------------


def _interactive_triage(records: list[dict], data_dir: Path) -> None:
    """Terminal-only interactive review. Raw entity text is shown to the user
    only on the local terminal; it is never written to any output the LLM sees.
    """
    from anon_proxy.crypto import resolve_key, decrypt_field
    from anon_proxy.retention import CorpusWriter

    key = resolve_key()
    corpus = CorpusWriter(data_dir)

    for rec in records:
        for span_ix, span in enumerate(rec.get("spans", [])):
            entity = (
                decrypt_field(span["enc_text"], key) if "enc_text" in span else "(no text)"
            )
            window = (
                decrypt_field(span["enc_window"], key) if "enc_window" in span else ""
            )
            print(
                f"\n[{rec['ts']}] {span['label']} via {span.get('source', '?')} "
                f"(score={span.get('score', '?')})"
            )
            print(f"  entity: {entity!r}")
            if window:
                print(f"  window: ...{window}...")
            choice = input("  [k]eep  [d]rop  [s]kip  [q]uit > ").strip().lower()
            if choice == "q":
                return
            if choice == "k":
                label = (
                    input("  canonical label (PERSON/EMAIL/.../NOT_PII): ")
                    .strip()
                    .upper()
                    or span["label"]
                )
                corpus.write({
                    **rec,
                    "review": {
                        "label": label,
                        "decision": "keep",
                        "reviewer_ts": datetime.now(timezone.utc).isoformat(),
                        "span_index": span_ix,
                    },
                })
            elif choice == "d":
                # Audit trail: record the drop decision but strip raw text
                corpus.write({
                    **{k: v for k, v in rec.items() if k != "enc_text"},
                    "spans": [
                        {k: v for k, v in s.items() if k not in ("enc_text", "enc_window")}
                        for s in rec.get("spans", [])
                    ],
                    "review": {
                        "decision": "drop",
                        "reviewer_ts": datetime.now(timezone.utc).isoformat(),
                        "span_index": span_ix,
                    },
                })
            # "s" or anything else: skip, continue to next record


_HANDLERS: dict[str, object] = {
    "triage": _triage_handler,
    "label": _stub,       # implemented in Phase 8B
    "corpus": _stub,      # implemented in Phase 8B
    "purge": _stub,       # implemented in Phase 8B
    "metrics": _metrics_handler,
    "export-key": _stub,  # implemented in Phase 8B
    "import-key": _stub,  # implemented in Phase 8B
    "suggest-regex": _stub,  # implemented in Phase 8B
}


if __name__ == "__main__":
    main()
