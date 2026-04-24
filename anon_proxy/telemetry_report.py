"""Render a summary of a telemetry JSONL log.

Usage:
    uv run python -m anon_proxy.telemetry_report [--path PATH]
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
    boundary_zone = 0  # missed within first/last 100 chars of any chunk
    per_chunk_idx: Counter[int] = Counter()
    for r in records:
        for m in r.get("regex_missed", []):
            miss_labels[m["label"]] += 1
            miss_total += 1
            if not m.get("ml_within_50ch"):
                isolated += 1
            if m.get("nearest_ml_label") and m.get("nearest_ml_label").upper() == m["label"].upper():
                nearest_same_label += 1
            per_chunk_idx[m.get("chunk_idx", 0)] += 1
            # Positional zone near chunk boundary — req_chars * (pos_pct % (1/chunks))
            chunks = r.get("req_chunks", 1) or 1
            if chunks > 1:
                within_chunk = (m.get("pos_pct", 0) * chunks) % 1
                if within_chunk < 0.1 or within_chunk > 0.9:
                    boundary_zone += 1

    print(f"{path}\n")
    print(f"{n:,} requests")
    print(f"  avg request:   {total_chars // n if n else 0:,} chars, "
          f"{total_chunks / n if n else 0:.2f} chunks "
          f"({multi_chunk:,} multi-chunk, {100 * multi_chunk / n if n else 0:.0f}%)")
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
    print(f"  no detector span within 50ch (isolated):     {isolated:>5,} ({pct(isolated):>4.0f}%)")
    print(f"  nearest detector span had the same label:    {nearest_same_label:>5,} "
          f"({pct(nearest_same_label):>4.0f}%)  (suggests boundary / fragmentation)")
    if any(r.get("req_chunks", 1) > 1 for r in records):
        print(f"  in outer 10% of a chunk (possible boundary): {boundary_zone:>5,} ({pct(boundary_zone):>4.0f}%)")
    print()
    print("Interpretation:")
    if isolated / miss_total > 0.5:
        print("  → Most misses are isolated (no ML span nearby). Points at clue-less PII")
        print("    as the dominant failure mode — regex canary would close this gap.")
    if nearest_same_label and nearest_same_label / miss_total > 0.3:
        print("  → A sizeable fraction of misses have a same-label detector span nearby.")
        print("    Likely boundary fragmentation — worth tightening merge-gap config or")
        print("    running the baseline regex as a real detector, not just an observer.")


if __name__ == "__main__":
    main()
