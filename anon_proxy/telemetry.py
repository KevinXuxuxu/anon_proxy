"""Observer-only telemetry: measure how often a baseline regex set catches
PII that the ML detector (and any user-configured regexes) missed.

Writes one JSON record per mask() call to a local JSONL file. Records contain
only labels, lengths, and positions — never the original PII value, never a
slice of the request text. The whole point of this proxy is that raw PII does
not leave the box; the telemetry log must honor the same contract.

Typical read:

    $ uv run python -m anon_proxy.telemetry_report

The sink is best-effort: any failure during observe() is logged to stderr
and swallowed so a telemetry bug can never break masking.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from anon_proxy.privacy_filter import PIIEntity
from anon_proxy.regex_detector import RegexDetector, load_patterns

DEFAULT_PATH = Path.home() / ".anon-proxy" / "telemetry.jsonl"
DEFAULT_PATTERNS_PATH = Path(__file__).parent / "telemetry_patterns.json"
DEFAULT_MAX_BYTES = 10_000_000
NEARBY_CHARS = 50


def load_default_patterns() -> dict[str, str]:
    return load_patterns(DEFAULT_PATTERNS_PATH)


def default_detector() -> RegexDetector:
    return RegexDetector(load_default_patterns())


class JSONLWriter:
    """Append-only writer with size-based rotation.

    Rotates to `<path>.1` when the next record would push the file past
    `max_bytes`. Only one rotation is kept; older history is dropped. This is
    a telemetry log, not an audit log — losing old samples is acceptable.
    """

    def __init__(self, path: Path | str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def __call__(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._rotate_if_needed(len(line.encode("utf-8")))
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self._max_bytes:
            return
        rotated = self._path.with_suffix(self._path.suffix + ".1")
        if rotated.exists():
            rotated.unlink()
        self._path.rename(rotated)


class TelemetryObserver:
    """Glue between Masker and a baseline detector + a sink.

    Masker calls `observe(text, ml_entities, extra_entities)` once per mask()
    invocation, passing the raw ML output and the user's configured regex
    output separately. The observer runs its baseline detector, computes
    which baseline spans were not caught by either pipeline, and hands a
    redacted record to the sink.
    """

    def __init__(
        self,
        detector,
        sink: Callable[[dict], None],
        chunk_size: int = 1500,
    ) -> None:
        self._detector = detector
        self._sink = sink
        self._chunk_size = chunk_size

    def observe(
        self,
        text: str,
        ml_entities: list[PIIEntity],
        extra_entities: list[PIIEntity],
    ) -> None:
        try:
            baseline = self._detector.detect(text)
            record = _build_record(
                text, ml_entities, extra_entities, baseline, self._chunk_size
            )
            self._sink(record)
        except Exception as exc:
            print(
                f"anon_proxy telemetry: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


def _build_record(
    text: str,
    ml_entities: list[PIIEntity],
    extra_entities: list[PIIEntity],
    baseline: list[PIIEntity],
    chunk_size: int,
) -> dict:
    detected = ml_entities + extra_entities
    missed = [
        _miss_record(r, detected, len(text), chunk_size)
        for r in baseline
        if not any(_overlaps(r, e) for e in detected)
    ]
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "req_chars": len(text),
        "req_chunks": max(1, (len(text) + chunk_size - 1) // chunk_size),
        "ml_spans": [_span(e) for e in ml_entities],
        "extra_spans": [_span(e) for e in extra_entities],
        "regex_missed": missed,
    }


def _span(e: PIIEntity) -> dict:
    return {"label": e.label, "len": e.end - e.start}


def _overlaps(a: PIIEntity, b: PIIEntity) -> bool:
    return a.start < b.end and b.start < a.end


def _miss_record(
    miss: PIIEntity,
    detected: list[PIIEntity],
    total_chars: int,
    chunk_size: int,
) -> dict:
    nearest = _nearest(miss, detected)
    nearest_dist = _distance(miss, nearest) if nearest is not None else None
    return {
        "label": miss.label,
        "len": miss.end - miss.start,
        "pos_pct": round(miss.start / max(1, total_chars), 3),
        "chunk_idx": miss.start // chunk_size,
        "ml_within_50ch": nearest_dist is not None and nearest_dist <= NEARBY_CHARS,
        "nearest_ml_label": nearest.label if nearest is not None else None,
    }


def _nearest(r: PIIEntity, detected: list[PIIEntity]) -> PIIEntity | None:
    if not detected:
        return None
    return min(detected, key=lambda e: _distance(r, e))


def _distance(a: PIIEntity, b: PIIEntity) -> int:
    if _overlaps(a, b):
        return 0
    return max(b.start - a.end, a.start - b.end)
