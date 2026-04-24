"""Tests for the telemetry observer.

No ML model is loaded here — observer/sink logic is tested against
hand-built PIIEntity spans so the suite runs in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anon_proxy.privacy_filter import PIIEntity
from anon_proxy.regex_detector import RegexDetector
from anon_proxy.telemetry import (
    JSONLWriter,
    TelemetryObserver,
    _build_record,
    _distance,
    _overlaps,
    load_default_patterns,
)


def _e(label: str, start: int, end: int, text: str = "x") -> PIIEntity:
    return PIIEntity(label=label, text=text, start=start, end=end, score=1.0)


# ---------- overlap / distance helpers ----------


def test_overlaps_true_when_spans_share_any_char():
    assert _overlaps(_e("A", 0, 5), _e("B", 4, 10))


def test_overlaps_false_when_spans_only_touch():
    # [0,5) and [5,10) touch at 5 but do not share a char.
    assert not _overlaps(_e("A", 0, 5), _e("B", 5, 10))


def test_distance_zero_for_overlap():
    assert _distance(_e("A", 0, 5), _e("B", 3, 8)) == 0


def test_distance_counts_gap():
    assert _distance(_e("A", 0, 5), _e("B", 10, 12)) == 5
    assert _distance(_e("A", 20, 30), _e("B", 5, 10)) == 10


# ---------- record building ----------


def test_record_has_no_pii_content():
    """A record must not contain any substring of the original text."""
    text = "email alice@company.com and phone 555-867-5309"
    ml = []  # detector saw nothing
    extra = []
    baseline = [
        _e("EMAIL", 6, 23, "alice@company.com"),
        _e("PHONE_NANP", 34, 46, "555-867-5309"),
    ]
    rec = _build_record(text, ml, extra, baseline, chunk_size=1500)
    blob = json.dumps(rec)
    assert "alice" not in blob
    assert "company.com" not in blob
    assert "555" not in blob
    assert "867" not in blob


def test_record_flags_all_baseline_as_missed_when_detectors_empty():
    text = "x" * 100
    baseline = [_e("EMAIL", 10, 20), _e("PHONE_NANP", 50, 62)]
    rec = _build_record(text, [], [], baseline, chunk_size=1500)
    assert len(rec["regex_missed"]) == 2
    assert {m["label"] for m in rec["regex_missed"]} == {"EMAIL", "PHONE_NANP"}


def test_record_does_not_flag_baseline_caught_by_ml():
    text = "x" * 100
    ml = [_e("private_email", 8, 22)]  # covers baseline span
    baseline = [_e("EMAIL", 10, 20)]
    rec = _build_record(text, ml, [], baseline, chunk_size=1500)
    assert rec["regex_missed"] == []


def test_record_does_not_flag_baseline_caught_by_extra_regex():
    """If the user has a working regex that already catches this, don't flag it."""
    text = "x" * 100
    extra = [_e("SSN", 10, 21)]
    baseline = [_e("SSN", 10, 21)]
    rec = _build_record(text, [], extra, baseline, chunk_size=1500)
    assert rec["regex_missed"] == []


def test_miss_record_encodes_position_and_chunk():
    text = "x" * 3000
    baseline = [_e("EMAIL", 1800, 1820)]
    rec = _build_record(text, [], [], baseline, chunk_size=1500)
    miss = rec["regex_missed"][0]
    assert miss["label"] == "EMAIL"
    assert miss["len"] == 20
    assert miss["chunk_idx"] == 1
    assert 0.59 <= miss["pos_pct"] <= 0.61  # 1800/3000 = 0.6


def test_ml_within_50ch_true_when_detector_span_adjacent():
    text = "x" * 200
    ml = [_e("private_person", 40, 60)]
    baseline = [_e("EMAIL", 90, 100)]  # 30ch gap from ml
    rec = _build_record(text, ml, [], baseline, chunk_size=1500)
    miss = rec["regex_missed"][0]
    assert miss["ml_within_50ch"] is True
    assert miss["nearest_ml_label"] == "private_person"


def test_ml_within_50ch_false_when_isolated():
    text = "x" * 500
    ml = [_e("private_person", 0, 10)]
    baseline = [_e("EMAIL", 400, 420)]
    rec = _build_record(text, ml, [], baseline, chunk_size=1500)
    miss = rec["regex_missed"][0]
    assert miss["ml_within_50ch"] is False
    assert miss["nearest_ml_label"] == "private_person"  # still reported, just far


def test_record_shape_has_required_fields():
    rec = _build_record("hello", [], [], [], chunk_size=1500)
    assert set(rec) >= {"ts", "req_chars", "req_chunks", "ml_spans", "extra_spans", "regex_missed"}
    assert rec["req_chars"] == 5
    assert rec["req_chunks"] == 1


def test_chunks_computed_from_chunk_size():
    rec = _build_record("x" * 3001, [], [], [], chunk_size=1500)
    assert rec["req_chunks"] == 3


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
    w({"x": "a" * 30})  # fits
    w({"x": "b" * 30})  # triggers rotation
    assert path.exists()
    rotated = path.with_suffix(".jsonl.1")
    assert rotated.exists()


# ---------- TelemetryObserver integration ----------


def test_observer_catches_detector_exceptions(tmp_path: Path, capsys):
    class Boom:
        def detect(self, text):
            raise RuntimeError("boom")

    out = tmp_path / "log.jsonl"
    obs = TelemetryObserver(detector=Boom(), sink=JSONLWriter(out), chunk_size=1500)
    # Must not raise; must print to stderr.
    obs.observe("hello", [], [])
    err = capsys.readouterr().err
    assert "boom" in err
    assert not out.exists()  # sink never called


def test_observer_writes_record_on_happy_path(tmp_path: Path):
    out = tmp_path / "log.jsonl"
    det = RegexDetector({"EMAIL": r"\b\w+@\w+\.\w+\b"})
    obs = TelemetryObserver(detector=det, sink=JSONLWriter(out), chunk_size=1500)
    obs.observe("reach me at alice@example.com", [], [])
    rec = json.loads(out.read_text().strip())
    assert rec["regex_missed"][0]["label"] == "EMAIL"
    # No original PII in the record:
    assert "alice" not in out.read_text()


# ---------- default patterns ----------


def test_default_patterns_compile():
    # load_patterns + RegexDetector will raise if any regex is invalid.
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
