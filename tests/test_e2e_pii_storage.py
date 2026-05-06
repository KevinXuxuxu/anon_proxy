"""End-to-end: server captures with PII storage → CLI triages → corpus grows.

This is the integration smoke test for the local-PII-storage feature. It
exercises every layer in sequence (Masker → TelemetryObserver → RawWriter →
encrypted on disk → triage CLI → CorpusWriter) to catch breakage that unit
tests miss."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from anon_proxy.crypto import generate_key, store_key, decrypt_field
from anon_proxy.pipeline import AttributedSpan
from anon_proxy.regex_detector import RegexDetector
from anon_proxy.retention import RetentionConfig, RawWriter, MetricsWriter
from anon_proxy.telemetry import CaptureMode, TelemetryObserver
from anon_proxy.triage_cli import main as triage_main


def test_e2e_capture_metadata_only_triage(tmp_path, monkeypatch, fake_keyring, capsys):
    """Capture an EMAIL via Lean mode → triage --json --metadata-only must
    surface the label/source/score WITHOUT decrypting (no key needed)."""
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    key = generate_key()
    store_key(key)

    # 1. Capture via the actual writer pipeline used in production
    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=30, raw_size_mb=50)
    metrics = MetricsWriter(tmp_path)
    raw = RawWriter(cfg, metrics_writer=metrics)
    detector = RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})
    obs = TelemetryObserver(
        detector=detector, sink=raw.write,
        capture_mode=CaptureMode.LEAN, encryption_key=key,
    )

    text = "Contact alice@example.com about the project."
    user_entities = detector.detect(text)
    spans = [AttributedSpan(entity=e, source="user_regex") for e in user_entities]
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[], user_spans=spans, kept=spans, events=[])

    # Verify on disk: encrypted field present, raw text NOT present anywhere
    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_content = raw_path.read_text()
    assert "alice@example.com" not in raw_content, "raw PII leaked to disk"
    assert "v1:" in raw_content, "encrypted field not written"

    # 2. Triage in --metadata-only mode (no keychain access needed)
    capsys.readouterr()  # drain anything emitted during capture
    triage_main(["triage", "--json", "--metadata-only", "--days", "365"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert len(parsed) == 1
    span = parsed[0]["spans"][0]
    assert span["label"] == "EMAIL"
    assert span["source"] == "user_regex"
    # Encrypted fields stripped
    assert "enc_text" not in span
    assert "enc_window" not in span


def test_e2e_capture_interactive_triage_promotes_to_corpus(tmp_path, monkeypatch, fake_keyring, capsys):
    """Capture → interactive triage with stdin = 'k\\nEMAIL\\nq\\n' → corpus
    has the labeled record with decision=keep, label=EMAIL."""
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    key = generate_key()
    store_key(key)

    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=30, raw_size_mb=50)
    metrics = MetricsWriter(tmp_path)
    raw = RawWriter(cfg, metrics_writer=metrics)
    detector = RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})
    obs = TelemetryObserver(
        detector=detector, sink=raw.write,
        capture_mode=CaptureMode.LEAN, encryption_key=key,
    )

    text = "Email me at alice@example.com today."
    user_entities = detector.detect(text)
    spans = [AttributedSpan(entity=e, source="user_regex") for e in user_entities]
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[], user_spans=spans, kept=spans, events=[])

    # Drain capture output
    capsys.readouterr()

    # Interactive triage: keep this record, label it EMAIL, then quit
    monkeypatch.setattr(sys, "stdin", io.StringIO("k\nEMAIL\nq\n"))
    triage_main(["triage", "--days", "365"])

    corpus_path = tmp_path / "corpus.jsonl"
    assert corpus_path.exists(), "corpus.jsonl not created"
    lines = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    review = lines[0]["review"]
    assert review["decision"] == "keep"
    assert review["label"] == "EMAIL"
    # Verify the original encrypted entity text is still in the corpus record
    span_enc = lines[0]["spans"][0].get("enc_text")
    assert span_enc and span_enc.startswith("v1:")
    assert decrypt_field(span_enc, key) == "alice@example.com"


def test_e2e_zero_pii_mode_disk_has_no_pii(tmp_path, monkeypatch, capsys):
    """Sanity: ZERO_PII mode never writes PII content to disk, never asks for a key."""
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)

    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=30, raw_size_mb=50)
    metrics = MetricsWriter(tmp_path)
    raw = RawWriter(cfg, metrics_writer=metrics)
    detector = RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})
    obs = TelemetryObserver(
        detector=detector, sink=raw.write,
        capture_mode=CaptureMode.ZERO_PII,
        encryption_key=None,
    )

    text = "Reach me at bob@acme.com for details."
    user_entities = detector.detect(text)
    spans = [AttributedSpan(entity=e, source="user_regex") for e in user_entities]
    with obs.new_batch() as batch:
        batch.observe_v2(text=text, ml_spans=[], user_spans=spans, kept=spans, events=[])

    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_content = raw_path.read_text()
    assert "bob@acme.com" not in raw_content
    assert "v1:" not in raw_content  # no encryption attempted in ZERO_PII
    # Label is still cleartext (that's by design)
    assert "EMAIL" in raw_content
