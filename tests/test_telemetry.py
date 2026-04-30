"""Tests for the telemetry observer + batch.

No ML model is loaded here — observer logic is tested against hand-built
PIIEntity spans so the suite runs in milliseconds.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from anon_proxy.adapters import anthropic as anthropic_adapter
from anon_proxy.masker import Masker
from anon_proxy.privacy_filter import PIIEntity, _split_chunks
from anon_proxy.regex_detector import RegexDetector
from anon_proxy.telemetry import (
    JSONLWriter,
    TelemetryBatch,
    TelemetryObserver,
    _distance,
    _in_boundary_zone,
    _overlaps,
    _resolve_baseline_overlaps,
    default_detector,
    load_default_patterns,
)


def _e(label: str, start: int, end: int, text: str = "x", score: float = 1.0) -> PIIEntity:
    return PIIEntity(label=label, text=text, start=start, end=end, score=score)


# ---------- overlap / distance helpers ----------


def test_overlaps_true_when_spans_share_any_char():
    assert _overlaps(_e("A", 0, 5), _e("B", 4, 10))


def test_overlaps_false_when_spans_only_touch():
    assert not _overlaps(_e("A", 0, 5), _e("B", 5, 10))


def test_distance_zero_for_overlap():
    assert _distance(_e("A", 0, 5), _e("B", 3, 8)) == 0


def test_distance_counts_gap():
    assert _distance(_e("A", 0, 5), _e("B", 10, 12)) == 5
    assert _distance(_e("A", 20, 30), _e("B", 5, 10)) == 10


# ---------- baseline overlap collapsing (fix #2) ----------


def test_resolve_baseline_keeps_longer_of_overlapping_duplicates():
    # Same value matched by two patterns: shorter one gets dropped.
    spans = [_e("PHONE_NANP", 10, 22), _e("PHONE_LOOSE", 10, 22)]
    kept = _resolve_baseline_overlaps(spans)
    assert len(kept) == 1


def test_resolve_baseline_preserves_disjoint_spans():
    spans = [_e("EMAIL", 0, 10), _e("PHONE", 50, 62)]
    kept = _resolve_baseline_overlaps(spans)
    assert len(kept) == 2


def test_resolve_baseline_prefers_longer_span():
    # Loose pattern matches a wider span than strict — keep the longer one.
    spans = [_e("STRICT", 10, 22), _e("LOOSE", 5, 25)]
    kept = _resolve_baseline_overlaps(spans)
    assert len(kept) == 1
    assert kept[0].label == "LOOSE"


# ---------- boundary zone (fix #3) ----------


def test_in_boundary_zone_false_for_single_chunk():
    offsets = [(0, 1500)]
    assert not _in_boundary_zone(_e("EMAIL", 100, 120), offsets)


def test_in_boundary_zone_true_near_internal_boundary():
    offsets = [(0, 1000), (1000, 2000)]
    assert _in_boundary_zone(_e("EMAIL", 990, 1005), offsets)  # straddles
    assert _in_boundary_zone(_e("EMAIL", 960, 975), offsets)   # just before
    assert _in_boundary_zone(_e("EMAIL", 1030, 1045), offsets) # just after


def test_in_boundary_zone_false_far_from_internal_boundary():
    offsets = [(0, 1000), (1000, 2000)]
    assert not _in_boundary_zone(_e("EMAIL", 300, 320), offsets)


def test_in_boundary_zone_ignores_text_edges():
    # Only INTERNAL boundaries count; edges at text start/end are not splits.
    offsets = [(0, 1000), (1000, 2000)]
    assert not _in_boundary_zone(_e("EMAIL", 0, 10), offsets)
    assert not _in_boundary_zone(_e("EMAIL", 1990, 2000), offsets)


# ---------- batch accumulation (fix #1) ----------


class _FakeObserver:
    def __init__(self, detector, chunker=None):
        self._detector = detector
        self._chunker = chunker
        self._encryption_key = None
        self.records: list[dict] = []

    def baseline(self, text):
        return self._detector.detect(text)

    def chunks(self, text):
        return self._chunker(text) if self._chunker else []

    def stores_pii(self) -> bool:
        return False

    def stores_full_text(self) -> bool:
        return False

    def _write(self, record):
        self.records.append(record)


def test_batch_commits_one_record_for_multiple_observes():
    det = RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"})
    obs = _FakeObserver(det)
    batch = TelemetryBatch(obs)
    batch.observe("first leaf text", [_e("private_person", 0, 5)], [])
    batch.observe("second leaf with email alice@foo.com", [], [])
    batch.commit()
    assert len(obs.records) == 1
    rec = obs.records[0]
    assert rec["req_chars"] == len("first leaf text") + len("second leaf with email alice@foo.com")
    assert len(rec["ml_spans"]) == 1
    assert len(rec["regex_missed"]) == 1
    assert rec["regex_missed"][0]["label"] == "EMAIL"


def test_batch_commit_is_idempotent():
    obs = _FakeObserver(RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}))
    batch = TelemetryBatch(obs)
    batch.observe("alice@foo.com", [], [])
    batch.commit()
    batch.commit()
    assert len(obs.records) == 1


def test_batch_as_context_manager():
    obs = _FakeObserver(RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}))
    with TelemetryBatch(obs) as batch:
        batch.observe("bob@bar.com", [], [])
    assert len(obs.records) == 1


def test_batch_skips_empty_record():
    obs = _FakeObserver(RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}))
    batch = TelemetryBatch(obs)
    batch.commit()
    assert obs.records == []


def test_batch_deduplicates_overlapping_baseline_hits():
    """PHONE_NANP and PHONE_LOOSE will both fire on '555-867-5309' with the
    shipped defaults — miss count must still be 1."""
    det = RegexDetector(load_default_patterns())
    obs = _FakeObserver(det)
    batch = TelemetryBatch(obs)
    batch.observe("Call 555-867-5309 later", [], [])  # both ML and user regex empty
    batch.commit()
    rec = obs.records[0]
    assert len(rec["regex_missed"]) == 1  # NOT 2


def test_batch_uses_chunker_for_boundary_zone():
    def chunker(text):
        return _split_chunks_ranges(text, 1000)

    det = RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"})
    obs = _FakeObserver(det, chunker=chunker)
    # Email straddling chunk boundary at char 1000.
    long_text = ("x " * 495) + "alice@foo.com" + (" x" * 495)
    batch = TelemetryBatch(obs)
    batch.observe(long_text, [], [])
    batch.commit()
    rec = obs.records[0]
    miss = rec["regex_missed"][0]
    assert miss["multi_chunk_leaf"] is True
    # The email is near the midpoint (~1000 chars in) which should be
    # near a chunk boundary given chunk_size=1000.
    assert miss["boundary_zone"] is True


def _split_chunks_ranges(text: str, size: int) -> list[tuple[int, int]]:
    return [(off, off + len(ch)) for off, ch in _split_chunks(text, size)]


# ---------- record content: no PII leakage ----------


def test_record_has_no_pii_content():
    text = "email alice@company.com and phone 555-867-5309"
    det = RegexDetector(load_default_patterns())
    obs = _FakeObserver(det)
    batch = TelemetryBatch(obs)
    batch.observe(text, [], [])
    batch.commit()
    blob = json.dumps(obs.records)
    assert "alice" not in blob
    assert "company.com" not in blob
    assert "555" not in blob
    assert "867" not in blob


# ---------- Observer direct (single-shot) ----------


def test_observer_single_shot_writes_one_record(tmp_path: Path):
    out = tmp_path / "log.jsonl"
    det = RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"})
    obs = TelemetryObserver(detector=det, sink=JSONLWriter(out))
    obs.observe("reach me at alice@example.com", [], [])
    rec = json.loads(out.read_text().strip())
    assert rec["regex_missed"][0]["label"] == "EMAIL"
    assert "alice" not in out.read_text()


def test_observer_swallows_detector_exceptions(tmp_path: Path, capsys):
    class Boom:
        def detect(self, text):
            raise RuntimeError("boom")

    out = tmp_path / "log.jsonl"
    obs = TelemetryObserver(detector=Boom(), sink=JSONLWriter(out))
    obs.observe("hello", [], [])
    err = capsys.readouterr().err
    assert "boom" in err
    assert not out.exists()


# ---------- Masker + scope integration (fix #1) ----------


class _StubFilter:
    def detect(self, text):
        return []

    def chunk_ranges(self, text):
        return [(0, len(text))]


def test_request_scope_produces_one_record_per_request(tmp_path: Path):
    out = tmp_path / "log.jsonl"
    obs = TelemetryObserver(
        detector=RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}),
        sink=JSONLWriter(out),
        chunker=_StubFilter().chunk_ranges,
    )
    masker = Masker(filter=_StubFilter(), telemetry=obs)
    # Simulate the adapter calling mask() multiple times for one request:
    with masker.request_scope():
        masker.mask("first leaf")
        masker.mask("leaf with alice@foo.com")
        masker.mask("third leaf with bob@bar.com")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1, f"expected 1 record, got {len(lines)}"
    rec = json.loads(lines[0])
    assert len(rec["regex_missed"]) == 2


def test_request_scope_is_async_safe(tmp_path: Path):
    """Two concurrent async tasks must not cross-contaminate batches."""
    out = tmp_path / "log.jsonl"
    obs = TelemetryObserver(
        detector=RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}),
        sink=JSONLWriter(out),
        chunker=_StubFilter().chunk_ranges,
    )
    masker = Masker(filter=_StubFilter(), telemetry=obs)

    async def one_request(text: str):
        with masker.request_scope():
            masker.mask(text)
            await asyncio.sleep(0)  # yield, letting the other task run
            masker.mask("and " + text)

    async def drive():
        await asyncio.gather(
            one_request("alice@a.com"),
            one_request("bob@b.com"),
        )

    asyncio.run(drive())
    recs = [json.loads(ln) for ln in out.read_text().strip().splitlines()]
    assert len(recs) == 2
    # Each record should have 2 misses (both leaves each), not 4.
    for r in recs:
        assert len(r["regex_missed"]) == 2


def test_mask_without_scope_still_emits_per_call(tmp_path: Path):
    """Backward-compat: direct Masker.mask() outside a scope still records."""
    out = tmp_path / "log.jsonl"
    obs = TelemetryObserver(
        detector=RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}),
        sink=JSONLWriter(out),
        chunker=_StubFilter().chunk_ranges,
    )
    masker = Masker(filter=_StubFilter(), telemetry=obs)
    masker.mask("alice@a.com")
    masker.mask("bob@b.com")
    assert len(out.read_text().strip().splitlines()) == 2


# ---------- Anthropic adapter integration (fix #1) ----------


def test_adapter_mask_request_emits_one_record_per_payload(tmp_path: Path):
    out = tmp_path / "log.jsonl"
    obs = TelemetryObserver(
        detector=RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"}),
        sink=JSONLWriter(out),
        chunker=_StubFilter().chunk_ranges,
    )
    masker = Masker(filter=_StubFilter(), telemetry=obs)
    body = {
        "model": "claude-3",
        "messages": [
            {"role": "user", "content": "reach me at alice@foo.com"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok, contacting bob@bar.com"},
                    {
                        "type": "tool_use",
                        "name": "email",
                        "input": {"to": "carol@baz.com", "subject": "hi"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": "sent to dave@qux.com",
                    }
                ],
            },
        ],
    }
    # proxy now owns the scope; callers outside the proxy must wrap their own
    with masker.request_scope():
        anthropic_adapter.mask_request(body, masker)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1, f"expected 1 record per request, got {len(lines)}"
    rec = json.loads(lines[0])
    # Four distinct emails, each in a different leaf, all missed by ML (no filter).
    assert len(rec["regex_missed"]) == 4


# ---------- JSONLWriter ----------


def test_writer_appends_one_line_per_call(tmp_path: Path):
    w = JSONLWriter(tmp_path / "log.jsonl")
    w({"a": 1})
    w({"a": 2})
    lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    assert [json.loads(x) for x in lines] == [{"a": 1}, {"a": 2}]


def test_writer_creates_parent_dir(tmp_path: Path):
    w = JSONLWriter(tmp_path / "nested" / "dir" / "log.jsonl")
    w({"ok": True})
    assert (tmp_path / "nested" / "dir" / "log.jsonl").exists()


def test_writer_rotates_when_over_limit(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    w = JSONLWriter(path, max_bytes=50)
    w({"x": "a" * 30})
    w({"x": "b" * 30})
    assert path.exists()
    assert path.with_suffix(".jsonl.1").exists()


# ---------- default patterns ----------


def test_default_patterns_compile():
    RegexDetector(load_default_patterns())


def test_default_patterns_catch_common_shapes():
    det = RegexDetector(load_default_patterns())
    cases = {
        "Reach me at alice@example.com.": "EMAIL",
        "Phone: 555-867-5309": "PHONE_NANP",
        "SSN 123-45-6789 on file": "SSN",
        "Server at 10.0.0.1 is down": "IPV4",
    }
    for text, expected_label in cases.items():
        labels = {e.label for e in det.detect(text)}
        assert expected_label in labels, f"expected {expected_label} in {labels} for: {text!r}"


from anon_proxy.mapping import PIIStore
from anon_proxy.pipeline import AttributedSpan, OverlapEvent


class _DummyFilter:
    def detect(self, text):
        return []

    def chunk_ranges(self, text):
        return [(0, len(text))]


def _aspan(label: str, start: int, end: int, source: str, score: float = 1.0) -> AttributedSpan:
    return AttributedSpan(
        entity=PIIEntity(label=label, text="x" * (end - start), start=start, end=end, score=score),
        source=source,
    )


def test_v2_record_has_schema_field(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    batch.observe_v2(
        text="hello",
        ml_spans=[_aspan("private_email", 0, 5, "ml")],
        user_spans=[],
        kept=[_aspan("private_email", 0, 5, "ml")],
        events=[],
    )
    batch.commit()
    rec = json.loads(log.read_text().strip())
    assert rec["schema"] == 3
    assert "spans" in rec
    assert "overlap_events" in rec
    assert "chunks" in rec


def test_v2_record_attributes_each_span(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    ml = _aspan("private_email", 0, 5, "ml", 0.95)
    rx = _aspan("EMAIL", 0, 5, "user_regex", 1.0)
    # rx wins the score tie because it has score 1.0 vs ml's 0.95
    batch.observe_v2(
        text="hello",
        ml_spans=[ml],
        user_spans=[rx],
        kept=[rx],
        events=[OverlapEvent(winner=rx, loser=ml, reason="overlap_score_tie")],
    )
    batch.commit()
    rec = json.loads(log.read_text().strip())
    sources = {s["source"] for s in rec["spans"] if s["source"] != "baseline"}
    assert sources == {"ml", "user_regex"}
    kept_flags = {(s["source"], s["kept"]) for s in rec["spans"] if s["source"] != "baseline"}
    assert kept_flags == {("ml", False), ("user_regex", True)}
    span_by_source = {s["source"]: s for s in rec["spans"] if s["source"] != "baseline"}
    assert span_by_source["user_regex"]["reason"] == "overlap_score_tie"  # kept winner
    assert span_by_source["ml"]["reason"] == "overlap_score_tie"          # loser
    assert rec["overlap_events"] == [
        {
            "winner_source": "user_regex",
            "loser_source": "ml",
            "winner_label": "EMAIL",
            "loser_label": "private_email",
            "reason": "overlap_score_tie",
        }
    ]


def test_v2_keeps_v1_fields_for_backwards_compat(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    ml = _aspan("private_email", 0, 5, "ml")
    batch.observe_v2(text="hello", ml_spans=[ml], user_spans=[], kept=[ml], events=[])
    batch.commit()
    rec = json.loads(log.read_text().strip())
    assert rec["ml_spans"] == [{"label": "private_email", "len": 5}]
    assert rec["extra_spans"] == []
    assert "regex_missed" in rec


def test_v2_chunks_field_counts_spans_per_chunk(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    chunker = lambda t: [(0, 10), (10, len(t))] if len(t) > 10 else [(0, len(t))]
    obs = TelemetryObserver(default_detector(), JSONLWriter(log), chunker=chunker)
    batch = obs.new_batch()
    ml1 = _aspan("private_email", 2, 7, "ml")
    ml2 = _aspan("private_phone", 12, 18, "ml")
    batch.observe_v2(
        text="x" * 20, ml_spans=[ml1, ml2], user_spans=[], kept=[ml1, ml2], events=[]
    )
    batch.commit()
    rec = json.loads(log.read_text().strip())
    assert rec["chunks"] == [
        {"chars": 10, "ml_spans": 1, "user_spans": 0, "baseline_spans": 0},
        {"chars": 10, "ml_spans": 1, "user_spans": 0, "baseline_spans": 0},
    ]


def test_v2_no_pii_content_in_record(tmp_path: Path):
    """Same invariant as v1: raw input substrings must not appear in the log."""
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    secret = "hunter2hunter2"
    ml = _aspan("private_password", 0, len(secret), "ml")
    batch.observe_v2(
        text=f"my password is {secret}!",
        ml_spans=[ml],
        user_spans=[],
        kept=[ml],
        events=[],
    )
    batch.commit()
    blob = log.read_text()
    assert secret not in blob
    assert "hunter" not in blob


# ---------- request_scope reentrancy (Task 1) ----------


def test_request_scope_yields_batch(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    masker = Masker(filter=_DummyFilter(), store=PIIStore(), telemetry=obs)
    with masker.request_scope() as batch:
        assert batch is not None
        assert hasattr(batch, "commit")
        assert hasattr(batch, "observe_v2")


def test_request_scope_is_reentrant(tmp_path: Path):
    """Nested scopes reuse the outer batch and don't commit early."""
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    masker = Masker(filter=_DummyFilter(), store=PIIStore(), telemetry=obs)
    with masker.request_scope() as outer:
        with masker.request_scope() as inner:
            assert inner is outer  # same batch instance
            masker.mask("hello")
        # After inner exits, the batch is NOT yet committed.
        assert (not log.exists()) or (log.read_text() == "")
    # Outer exit commits exactly once.
    text = log.read_text()
    assert text.count("\n") == 1


def test_request_scope_without_telemetry_yields_none(tmp_path: Path):
    masker = Masker(filter=_DummyFilter(), store=PIIStore(), telemetry=None)
    with masker.request_scope() as batch:
        assert batch is None


# ---------- record_latency (Task 2) ----------


def test_record_latency_appears_in_committed_record(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    batch.observe_v2(text="hi", ml_spans=[], user_spans=[], kept=[], events=[])
    batch.record_latency(mask_ms=3, upstream_ms=412, unmask_ms=2, total_ms=420)
    batch.commit()
    rec = json.loads(log.read_text().strip())
    assert rec["latency_ms"] == {"mask": 3, "upstream": 412, "unmask": 2, "total": 420}


def test_record_latency_optional(tmp_path: Path):
    """Records without a record_latency call have no latency_ms key."""
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    batch.observe_v2(text="hi", ml_spans=[], user_spans=[], kept=[], events=[])
    batch.commit()
    rec = json.loads(log.read_text().strip())
    assert "latency_ms" not in rec


def test_record_latency_last_call_wins(tmp_path: Path):
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    batch = obs.new_batch()
    batch.observe_v2(text="hi", ml_spans=[], user_spans=[], kept=[], events=[])
    batch.record_latency(mask_ms=1, upstream_ms=1, unmask_ms=1, total_ms=3)
    batch.record_latency(mask_ms=10, upstream_ms=20, unmask_ms=5, total_ms=35)
    batch.commit()
    rec = json.loads(log.read_text().strip())
    assert rec["latency_ms"] == {"mask": 10, "upstream": 20, "unmask": 5, "total": 35}


def test_request_scope_isolated_per_masker(tmp_path: Path):
    """Two Masker instances must not share each other's batches."""
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    obs_a = TelemetryObserver(default_detector(), JSONLWriter(log_a))
    obs_b = TelemetryObserver(default_detector(), JSONLWriter(log_b))
    masker_a = Masker(filter=_DummyFilter(), store=PIIStore(), telemetry=obs_a)
    masker_b = Masker(filter=_DummyFilter(), store=PIIStore(), telemetry=obs_b)

    with masker_a.request_scope() as batch_a:
        with masker_b.request_scope() as batch_b:
            # masker_b must NOT reuse masker_a's batch
            assert batch_b is not batch_a
            masker_b.mask("hello-b")
        # masker_b's scope committed → log_b should have one record
        assert log_b.exists()
        assert log_b.read_text().count("\n") == 1
        masker_a.mask("hello-a")
    # masker_a's scope committed → log_a should have one record
    assert log_a.exists()
    assert log_a.read_text().count("\n") == 1


# ---------- v3 schema: CaptureMode + encrypted fields ----------


import pytest

from anon_proxy.crypto import decrypt_field, generate_key
from anon_proxy.pipeline import AttributedSpan
from anon_proxy.telemetry import CaptureMode


class _NullRegexDetector:
    def detect(self, text):
        return []


def _entity(label, start, end, text, score=0.9):
    from anon_proxy.privacy_filter import PIIEntity
    return PIIEntity(label=label, text=text, start=start, end=end, score=score)


def test_v3_lean_mode_encrypts_entity_text_and_window():
    key = generate_key()
    sink = []
    obs = TelemetryObserver(
        detector=_NullRegexDetector(),
        sink=sink.append,
        chunker=None,
        capture_mode=CaptureMode.LEAN,
        encryption_key=key,
    )
    text = "Hello Alice Smith from 2026 — call her tomorrow."
    ml = AttributedSpan(entity=_entity("PERSON", 6, 17, "Alice Smith"), source="ml")
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[ml], user_spans=[], kept=[ml], events=[])
    rec = sink[-1]
    assert rec["schema"] == 3
    span = rec["spans"][0]
    assert "enc_text" in span
    assert "enc_window" in span
    assert decrypt_field(span["enc_text"], key) == "Alice Smith"
    window = decrypt_field(span["enc_window"], key)
    assert "Alice Smith" in window
    assert rec.get("enc_text") is None  # full text only in CORPUS mode


def test_v3_zero_pii_mode_records_no_encrypted_fields():
    sink = []
    obs = TelemetryObserver(
        detector=_NullRegexDetector(),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
        encryption_key=None,
    )
    text = "Email me at alice@example.com please"
    ml = AttributedSpan(entity=_entity("EMAIL", 12, 31, "alice@example.com"), source="ml")
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[ml], user_spans=[], kept=[ml], events=[])
    rec = sink[-1]
    span = rec["spans"][0]
    assert "enc_text" not in span
    assert "enc_window" not in span
    assert rec.get("enc_text") is None


def test_v3_corpus_mode_encrypts_full_text():
    key = generate_key()
    sink = []
    obs = TelemetryObserver(
        detector=_NullRegexDetector(),
        sink=sink.append,
        capture_mode=CaptureMode.CORPUS,
        encryption_key=key,
    )
    text = "Long prompt with many sensitive details about Alice Smith and Bob Jones."
    ml = AttributedSpan(entity=_entity("PERSON", 46, 57, "Alice Smith"), source="ml")
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[ml], user_spans=[], kept=[ml], events=[])
    rec = sink[-1]
    assert decrypt_field(rec["enc_text"], key) == text


def test_v3_lean_mode_requires_key():
    with pytest.raises(ValueError, match="encryption_key required"):
        TelemetryObserver(
            detector=_NullRegexDetector(),
            sink=lambda r: None,
            capture_mode=CaptureMode.LEAN,
            encryption_key=None,
        )


def test_v3_corpus_mode_concatenates_multi_leaf_request():
    """A single API request walks multiple text leaves; CORPUS captures all of them."""
    key = generate_key()
    sink = []
    obs = TelemetryObserver(
        detector=_NullRegexDetector(),
        sink=sink.append,
        capture_mode=CaptureMode.CORPUS,
        encryption_key=key,
    )
    with obs.new_batch() as batch:
        batch.observe_v2(text="Hello Alice", ml_spans=[], user_spans=[], kept=[], events=[])
        batch.observe_v2(text="Goodbye Bob",  ml_spans=[], user_spans=[], kept=[], events=[])
    rec = sink[-1]
    decrypted = decrypt_field(rec["enc_text"], key)
    assert "Hello Alice" in decrypted and "Goodbye Bob" in decrypted


# ---------- side field (Phase 5) ----------


def test_v3_side_field_only_present_when_non_default():
    sink = []
    obs = TelemetryObserver(
        detector=_NullRegexDetector(),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
        encryption_key=None,
    )
    text = "Email me at alice@example.com"
    ml = AttributedSpan(entity=_entity("EMAIL", 12, 29, "alice@example.com"), source="ml")
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[ml], user_spans=[], kept=[ml], events=[])  # default side
        batch.observe_v2(text=text, ml_spans=[ml], user_spans=[], kept=[ml], events=[], side="response")
    rec = sink[-1]
    user_spans = [s for s in rec["spans"] if "side" not in s or s.get("side") == "user"]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    assert len(user_spans) == 1 and "side" not in user_spans[0]
    assert len(response_spans) == 1 and response_spans[0]["side"] == "response"
