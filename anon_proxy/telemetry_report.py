"""Render a summary of a telemetry JSONL log.

Usage:
    uv run python -m anon_proxy.telemetry_report [--path PATH]

Each line in the log is one API request (one `POST /v1/messages`) — records
are aggregated across all mask() leaf calls within that request by the
`Masker.request_scope()` context manager.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from anon_proxy.telemetry import DEFAULT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="anon-proxy telemetry report — analyze local mask logs for detector gaps",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to the telemetry JSONL log (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()

    path: Path = args.path
    if not path.exists():
        print(f"No telemetry log at {path}", file=sys.stderr)
        print(
            "Run the proxy with --telemetry to start collecting samples.",
            file=sys.stderr,
        )
        sys.exit(1)

    records = _load(path)
    if not records:
        print(f"{path}: log is empty.")
        return
    _render(records, path)


def _load(path: Path) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  skipping malformed line {i}: {e}", file=sys.stderr)
    return out


def _render(records: list[dict], path: Path) -> None:
    n = len(records)
    total_chars = sum(r.get("req_chars", 0) for r in records)
    total_chunks = sum(r.get("req_chunks", 1) for r in records)
    multi_chunk = sum(1 for r in records if r.get("req_chunks", 1) > 1)

    ml_labels: Counter[str] = Counter()
    extra_labels: Counter[str] = Counter()
    for r in records:
        ml_labels.update(s["label"] for s in r.get("ml_spans", []))
        extra_labels.update(s["label"] for s in r.get("extra_spans", []))

    miss_labels: Counter[str] = Counter()
    miss_total = 0
    isolated = 0
    nearest_same_label = 0
    boundary_zone_hits = 0
    for r in records:
        for m in r.get("regex_missed", []):
            miss_labels[m["label"]] += 1
            miss_total += 1
            if not m.get("ml_within_50ch"):
                isolated += 1
            nearest_label = m.get("nearest_ml_label")
            if nearest_label and nearest_label.upper() == m["label"].upper():
                nearest_same_label += 1
            if m.get("boundary_zone"):
                boundary_zone_hits += 1

    print(f"{path}\n")
    print(f"{n:,} API requests")
    print(
        f"  avg request:   {total_chars // n if n else 0:,} chars, "
        f"{total_chunks / n if n else 0:.2f} chunks "
        f"({multi_chunk:,} multi-chunk, {100 * multi_chunk / n if n else 0:.0f}%)"
    )
    print()

    total_ml = sum(ml_labels.values())
    print(f"ML detector: {total_ml:,} spans")
    for label, count in ml_labels.most_common():
        print(f"  {label:22s}  {count:>6,}")
    if extra_labels:
        total_extra = sum(extra_labels.values())
        print(f"\nUser regex detectors: {total_extra:,} spans")
        for label, count in extra_labels.most_common():
            print(f"  {label:22s}  {count:>6,}")
    print()

    if miss_total == 0:
        print("Regex baseline caught nothing that the configured detectors missed.")
        return

    print(f"Baseline regex caught but detectors missed: {miss_total:,} spans")
    for label, count in miss_labels.most_common():
        print(f"  {label:22s}  {count:>6,}")
    print()

    pct = lambda x: (100 * x / miss_total) if miss_total else 0.0
    print("Miss characterization:")
    print(f"  no detector span within 50ch (isolated):   {isolated:>5,} ({pct(isolated):>4.0f}%)")
    print(
        f"  nearest detector span same label:          {nearest_same_label:>5,} "
        f"({pct(nearest_same_label):>4.0f}%)  (suggests fragmentation)"
    )
    print(
        f"  within 50ch of a real chunk boundary:      {boundary_zone_hits:>5,} "
        f"({pct(boundary_zone_hits):>4.0f}%)  (suggests chunking cost)"
    )
    print()

    # --- Overlap events (v2 records only) ---
    overlap_pairs: Counter[tuple[str, str, str]] = Counter()
    for r in records:
        for ev in r.get("overlap_events", []):
            overlap_pairs[(ev["winner_source"], ev["loser_source"], ev["reason"])] += 1
    if overlap_pairs:
        print("Overlap events (winner ← loser, reason):")
        for (w, l, reason), count in overlap_pairs.most_common():
            print(f"  {w:>10s} ← {l:<10s}  {reason:<20s}  {count:>6,}")
        print()

    # --- Per-chunk distribution (v2 records only) ---
    chunk_count = 0
    chunk_ml = 0
    chunk_baseline = 0
    silent_chunks = 0  # chunks with baseline hits but zero ml hits
    for r in records:
        for c in r.get("chunks", []):
            chunk_count += 1
            chunk_ml += c.get("ml_spans", 0)
            chunk_baseline += c.get("baseline_spans", 0)
            if c.get("ml_spans", 0) == 0 and c.get("baseline_spans", 0) > 0:
                silent_chunks += 1
    if chunk_count:
        print("Per-chunk distribution:")
        print(f"  chunks total:                  {chunk_count:>6,}")
        print(f"  avg ml spans/chunk:            {chunk_ml / chunk_count:>6.2f}")
        print(f"  avg baseline spans/chunk:      {chunk_baseline / chunk_count:>6.2f}")
        print(
            f"  ml-silent chunks w/ baseline:  {silent_chunks:>6,} "
            f"({100 * silent_chunks / chunk_count:>4.0f}%)"
        )
        print()

    print("Interpretation:")
    if isolated / miss_total > 0.5:
        print("  → Most misses are isolated (no ML span nearby). Points at clue-less PII")
        print("    as the dominant failure mode — regex canary would close this gap.")
    if nearest_same_label and nearest_same_label / miss_total > 0.3:
        print("  → A sizeable fraction of misses have a same-label detector span nearby.")
        print("    Likely boundary fragmentation — worth tightening merge-gap config or")
        print("    running the baseline regex as a real detector, not just an observer.")
    if boundary_zone_hits and boundary_zone_hits / miss_total > 0.3:
        print("  → Many misses cluster near real chunk boundaries — raise --chunk-size")
        print("    or overlap chunks to give the model more context per pass.")


if __name__ == "__main__":
    main()
