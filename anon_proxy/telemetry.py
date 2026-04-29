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
from enum import Enum
from pathlib import Path

from anon_proxy.crypto import encrypt_field
from anon_proxy.pipeline import AttributedSpan, OverlapEvent
from anon_proxy.privacy_filter import PIIEntity
from anon_proxy.regex_detector import RegexDetector, load_patterns

DEFAULT_PATH = Path.home() / ".anon-proxy" / "telemetry.jsonl"
DEFAULT_PATTERNS_PATH = Path(__file__).parent / "telemetry_patterns.json"
DEFAULT_MAX_BYTES = 10_000_000
NEARBY_CHARS = 50
BOUNDARY_ZONE_CHARS = 50
WINDOW_CHARS = 200


class CaptureMode(Enum):
    ZERO_PII = "zero_pii"
    LEAN = "lean"
    CORPUS = "corpus"
    CORPUS_WITH_RESPONSES = "corpus_with_responses"

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

    v2 (preferred): callers invoke `observe_v2(text, ml_spans, user_spans, kept, events)`
    with `AttributedSpan` instances. The committed record carries `spans[]`
    with per-span attribution, `overlap_events[]`, and per-chunk counts.

    v1 (legacy): `observe(text, ml_entities, extra_entities)` still works for
    callers that haven't been migrated; the v2 fields are derived as best as
    possible (no overlap events available, kept inferred as union).

    v1 record fields (`ml_spans`, `extra_spans`, `regex_missed`) remain
    populated in v2 records so existing readers keep working.
    """

    __slots__ = (
        "_observer",
        "_spans",
        "_overlap_events",
        "_chunks_meta",
        "_missed",
        "_req_chars",
        "_req_chunks",
        "_committed",
        "_latency_ms",
        "_owner",
        "_raw_text_parts",
        "_enc_full_text",
    )

    def __init__(self, observer: "TelemetryObserver") -> None:
        self._observer = observer
        self._spans: list[dict] = []
        self._overlap_events: list[dict] = []
        self._chunks_meta: list[dict] = []
        self._missed: list[dict] = []
        self._req_chars: int = 0
        self._req_chunks: int = 0
        self._committed: bool = False
        self._latency_ms: dict | None = None
        self._owner: object | None = None
        self._raw_text_parts: list[str] = []
        self._enc_full_text: str | None = None

    def observe(
        self,
        text: str,
        ml_entities: list[PIIEntity],
        extra_entities: list[PIIEntity],
    ) -> None:
        """Legacy v1 signature — derives v2 attributes from raw entity lists."""
        ml_spans = [AttributedSpan(entity=e, source="ml") for e in ml_entities]
        user_spans = [AttributedSpan(entity=e, source="user_regex") for e in extra_entities]
        kept = ml_spans + user_spans
        self.observe_v2(text=text, ml_spans=ml_spans, user_spans=user_spans, kept=kept, events=[])

    def observe_v2(
        self,
        *,
        text: str,
        ml_spans: list[AttributedSpan],
        user_spans: list[AttributedSpan],
        kept: list[AttributedSpan],
        events: list[OverlapEvent],
        side: str = "user",
    ) -> None:
        try:
            self._accumulate(text, ml_spans, user_spans, kept, events, side=side)
        except Exception as exc:
            print(
                f"anon_proxy telemetry: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def record_latency(
        self,
        *,
        mask_ms: int,
        upstream_ms: int,
        unmask_ms: int,
        total_ms: int,
    ) -> None:
        """Record per-phase proxy latency. Last call wins; optional — records
        without a record_latency call omit `latency_ms` from the committed
        record entirely.
        """
        self._latency_ms = {
            "mask": int(mask_ms),
            "upstream": int(upstream_ms),
            "unmask": int(unmask_ms),
            "total": int(total_ms),
        }

    def _accumulate(
        self,
        text: str,
        ml_spans: list[AttributedSpan],
        user_spans: list[AttributedSpan],
        kept: list[AttributedSpan],
        events: list[OverlapEvent],
        *,
        side: str = "user",
    ) -> None:
        chunk_offsets = self._observer.chunks(text)
        baseline = _resolve_baseline_overlaps(self._observer.baseline(text))
        baseline_spans = [AttributedSpan(entity=e, source="baseline") for e in baseline]

        stores_pii = self._observer.stores_pii()
        text_for_enc = text if stores_pii else None
        key_for_enc = self._observer._encryption_key if stores_pii else None

        kept_ids = {id(s) for s in kept}
        for s in ml_spans + user_spans:
            self._spans.append(
                _span_record(s, kept=id(s) in kept_ids, events=events, text=text_for_enc, encryption_key=key_for_enc, side=side)
            )
        detected_entities = [s.entity for s in ml_spans + user_spans]
        for bs in baseline_spans:
            self._spans.append({
                "label": bs.label,
                "len": bs.length,
                "source": "baseline",
                "kept": False,
                "lost_to": None,
                "reason": "observer_only",
            })
            if any(_overlaps(bs.entity, e) for e in detected_entities):
                continue
            self._missed.append(_miss_record(bs.entity, detected_entities, chunk_offsets))

        for ev in events:
            self._overlap_events.append({
                "winner_source": ev.winner.source,
                "loser_source": ev.loser.source,
                "winner_label": ev.winner.label,
                "loser_label": ev.loser.label,
                "reason": ev.reason,
            })

        if chunk_offsets:
            for off, end in chunk_offsets:
                self._chunks_meta.append(_chunk_meta(off, end, ml_spans, user_spans, baseline_spans))
        else:
            self._chunks_meta.append(_chunk_meta(0, len(text), ml_spans, user_spans, baseline_spans))

        self._req_chars += len(text)
        self._req_chunks += len(chunk_offsets) if chunk_offsets else 1

        if self._observer.stores_full_text():
            self._raw_text_parts.append(text)

    def commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        if self._req_chars == 0 and not self._spans and not self._missed:
            return

        if self._observer.stores_full_text() and self._raw_text_parts:
            self._enc_full_text = encrypt_field(
                "\n---\n".join(self._raw_text_parts),
                self._observer._encryption_key,
            )

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "schema": 3,
            "req_chars": self._req_chars,
            "req_chunks": self._req_chunks,
            "spans": self._spans,
            "overlap_events": self._overlap_events,
            "chunks": self._chunks_meta,
            "regex_missed": self._missed,
            "ml_spans": [
                {"label": s["label"], "len": s["len"]}
                for s in self._spans
                if s["source"] == "ml" and s["kept"]
            ],
            "extra_spans": [
                {"label": s["label"], "len": s["len"]}
                for s in self._spans
                if s["source"] == "user_regex" and s["kept"]
            ],
            "enc_text": self._enc_full_text,
        }
        if self._latency_ms is not None:
            record["latency_ms"] = self._latency_ms
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
        *,
        capture_mode: CaptureMode = CaptureMode.ZERO_PII,
        encryption_key: bytes | None = None,
    ) -> None:
        if capture_mode != CaptureMode.ZERO_PII and encryption_key is None:
            raise ValueError(f"encryption_key required for capture_mode={capture_mode.value}")
        self._detector = detector
        self._sink = sink
        self._chunker = chunker
        self._capture_mode = capture_mode
        self._encryption_key = encryption_key

    def stores_pii(self) -> bool:
        return self._capture_mode != CaptureMode.ZERO_PII

    def stores_full_text(self) -> bool:
        return self._capture_mode in (CaptureMode.CORPUS, CaptureMode.CORPUS_WITH_RESPONSES)

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


def _span_record(
    span: AttributedSpan,
    *,
    kept: bool,
    events: list[OverlapEvent],
    text: str | None = None,
    encryption_key: bytes | None = None,
    side: str = "user",
) -> dict:
    """Build a v3 spans[] entry for an ml/user span (not baseline).

    When text and encryption_key are both provided (i.e. capture_mode != ZERO_PII),
    enc_text (the entity substring) and enc_window (surrounding context) are added.

    `side` is omitted from the record when it equals "user" (the default) to keep
    records compact; it is only present for non-default values (e.g. "response").
    """
    rec: dict = {
        "label": span.label,
        "len": span.length,
        "source": span.source,
        "kept": kept,
        "score": round(span.entity.score, 4),
    }
    if side != "user":
        rec["side"] = side
    # Check overlap events first — a span may be a winner (kept=True) or loser (kept=False).
    # Identity comparison (is) is safe because GreedyLongerWins.resolve doesn't copy spans;
    # OverlapEvent.winner / .loser refer to the same instances passed in via ml_spans/user_spans.
    found = False
    for ev in events:
        if ev.loser is span:
            rec["lost_to"] = ev.winner.source
            rec["reason"] = ev.reason
            found = True
            break
        if ev.winner is span:
            rec["lost_to"] = None
            rec["reason"] = ev.reason
            found = True
            break
    if not found:
        rec["lost_to"] = None
        rec["reason"] = "no_overlap"

    if text is not None and encryption_key is not None:
        entity_text = text[span.entity.start : span.entity.end]
        rec["enc_text"] = encrypt_field(entity_text, encryption_key)
        rec["enc_window"] = encrypt_field(_window_around(text, span, WINDOW_CHARS), encryption_key)

    return rec


def _window_around(text: str, span: AttributedSpan, chars: int) -> str:
    start = max(0, span.entity.start - chars)
    end = min(len(text), span.entity.end + chars)
    return text[start:end]


def _chunk_meta(
    offset: int,
    end: int,
    ml_spans: list[AttributedSpan],
    user_spans: list[AttributedSpan],
    baseline_spans: list[AttributedSpan],
) -> dict:
    return {
        "chars": end - offset,
        "ml_spans": sum(1 for s in ml_spans if offset <= s.start < end),
        "user_spans": sum(1 for s in user_spans if offset <= s.start < end),
        "baseline_spans": sum(1 for s in baseline_spans if offset <= s.start < end),
    }


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
