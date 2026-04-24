"""Observer-only telemetry: measure how often a baseline regex set catches
PII that the ML detector (and any user-configured regexes) missed.

One JSON record is written per *API request*, not per string leaf. The proxy's
Anthropic adapter walks every text block and tool-use input — a single
`/v1/messages` can trigger many `Masker.mask()` calls. Records are aggregated
within a `request_scope()` context and flushed as one line on exit.

Records contain only labels, lengths, and coarse positions — never the
original PII value, never a slice of the request text. The whole point of
this proxy is that raw PII does not leave the box; the telemetry log must
honor the same contract.

Typical read:

    $ uv run python -m anon_proxy.telemetry_report

Observer errors are logged to stderr and swallowed so a telemetry bug can
never break masking.
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
BOUNDARY_ZONE_CHARS = 50

ChunkFn = Callable[[str], list[tuple[int, int]]]


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


class TelemetryBatch:
    """Accumulates per-leaf observations across one API request.

    A batch is opened by `Masker.request_scope()` at the start of
    `anthropic_adapter.mask_request`, observes each `Masker.mask()` call
    inside the scope, and commits one record when the scope exits.

    Usable as a context manager; also commits on explicit `commit()`.
    """

    __slots__ = (
        "_observer",
        "_ml_spans",
        "_extra_spans",
        "_missed",
        "_req_chars",
        "_req_chunks",
        "_committed",
    )

    def __init__(self, observer: "TelemetryObserver") -> None:
        self._observer = observer
        self._ml_spans: list[dict] = []
        self._extra_spans: list[dict] = []
        self._missed: list[dict] = []
        self._req_chars: int = 0
        self._req_chunks: int = 0
        self._committed: bool = False

    def observe(
        self,
        text: str,
        ml_entities: list[PIIEntity],
        extra_entities: list[PIIEntity],
    ) -> None:
        try:
            self._accumulate(text, ml_entities, extra_entities)
        except Exception as exc:
            print(
                f"anon_proxy telemetry: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _accumulate(
        self,
        text: str,
        ml_entities: list[PIIEntity],
        extra_entities: list[PIIEntity],
    ) -> None:
        chunk_offsets = self._observer.chunks(text)
        detected = ml_entities + extra_entities
        baseline = _resolve_baseline_overlaps(self._observer.baseline(text))
        for r in baseline:
            if any(_overlaps(r, e) for e in detected):
                continue
            self._missed.append(_miss_record(r, detected, chunk_offsets))
        self._ml_spans.extend(_span(e) for e in ml_entities)
        self._extra_spans.extend(_span(e) for e in extra_entities)
        self._req_chars += len(text)
        self._req_chunks += len(chunk_offsets) if chunk_offsets else 1

    def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        if self._req_chars == 0 and not self._ml_spans and not self._missed:
            return
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "req_chars": self._req_chars,
            "req_chunks": self._req_chunks,
            "ml_spans": self._ml_spans,
            "extra_spans": self._extra_spans,
            "regex_missed": self._missed,
        }
        self._observer._write(record)

    def __enter__(self) -> "TelemetryBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.commit()


class TelemetryObserver:
    """Owns the baseline detector, the chunk source, and the sink.

    `chunker` returns the actual chunk offsets that `PrivacyFilter.detect()`
    uses, so miss records can say "this miss was 7 characters from an internal
    chunk boundary" without guessing. If omitted, chunk metadata degrades
    gracefully to single-chunk.
    """

    def __init__(
        self,
        detector,
        sink: Callable[[dict], None],
        chunker: ChunkFn | None = None,
    ) -> None:
        self._detector = detector
        self._sink = sink
        self._chunker = chunker

    def new_batch(self) -> TelemetryBatch:
        return TelemetryBatch(self)

    def observe(
        self,
        text: str,
        ml_entities: list[PIIEntity],
        extra_entities: list[PIIEntity],
    ) -> None:
        """Single-shot: accumulate + commit in one call.

        Used when no `request_scope` is active (e.g. unit tests). In the proxy
        server, the Anthropic adapter always wraps mask_request in a scope, so
        this path is not hit.
        """
        batch = self.new_batch()
        batch.observe(text, ml_entities, extra_entities)
        batch.commit()

    def baseline(self, text: str) -> list[PIIEntity]:
        return self._detector.detect(text)

    def chunks(self, text: str) -> list[tuple[int, int]]:
        if self._chunker is None:
            return []
        return self._chunker(text)

    def _write(self, record: dict) -> None:
        try:
            self._sink(record)
        except Exception as exc:
            print(
                f"anon_proxy telemetry: sink error {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


# ---------- helpers ----------


def _span(e: PIIEntity) -> dict:
    return {"label": e.label, "len": e.end - e.start}


def _overlaps(a: PIIEntity, b: PIIEntity) -> bool:
    return a.start < b.end and b.start < a.end


def _resolve_baseline_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep a non-overlapping subset of baseline spans, preferring longer / higher-score.

    The shipped pattern set deliberately includes loose + strict variants
    (e.g. PHONE_LOOSE, PHONE_NANP) so one real phone number fires twice. Without
    this pass, `regex_missed` would count the same leak multiple times.
    """
    if not entities:
        return entities
    ordered = sorted(
        entities,
        key=lambda e: (e.start, -(e.end - e.start), -e.score, e.label),
    )
    kept: list[PIIEntity] = []
    for e in ordered:
        if kept and e.start < kept[-1].end:
            prev = kept[-1]
            prev_len = prev.end - prev.start
            cur_len = e.end - e.start
            if cur_len > prev_len or (cur_len == prev_len and e.score > prev.score):
                kept[-1] = e
            continue
        kept.append(e)
    return kept


def _miss_record(
    miss: PIIEntity,
    detected: list[PIIEntity],
    chunk_offsets: list[tuple[int, int]],
) -> dict:
    nearest = _nearest(miss, detected)
    nearest_dist = _distance(miss, nearest) if nearest is not None else None
    return {
        "label": miss.label,
        "len": miss.end - miss.start,
        "ml_within_50ch": nearest_dist is not None and nearest_dist <= NEARBY_CHARS,
        "nearest_ml_label": nearest.label if nearest is not None else None,
        "boundary_zone": _in_boundary_zone(miss, chunk_offsets),
        "multi_chunk_leaf": len(chunk_offsets) > 1,
    }


def _nearest(r: PIIEntity, detected: list[PIIEntity]) -> PIIEntity | None:
    if not detected:
        return None
    return min(detected, key=lambda e: _distance(r, e))


def _distance(a: PIIEntity, b: PIIEntity) -> int:
    if _overlaps(a, b):
        return 0
    return max(b.start - a.end, a.start - b.end)


def _in_boundary_zone(miss: PIIEntity, chunk_offsets: list[tuple[int, int]]) -> bool:
    """True if the miss sits within BOUNDARY_ZONE_CHARS of an internal chunk boundary.

    Only INTERNAL boundaries count — the start of the first chunk and the end
    of the last chunk are the text's own edges, not splits introduced by the
    chunker. A single-chunk leaf has no internal boundaries, so this returns
    False.
    """
    if len(chunk_offsets) <= 1:
        return False
    for i in range(len(chunk_offsets) - 1):
        boundary = chunk_offsets[i][1]
        if (
            abs(miss.start - boundary) < BOUNDARY_ZONE_CHARS
            or abs(miss.end - boundary) < BOUNDARY_ZONE_CHARS
        ):
            return True
    return False
