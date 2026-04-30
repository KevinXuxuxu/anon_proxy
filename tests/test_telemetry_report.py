"""Tests for Phase 9 report extensions: trend, leak-back, storage health, --with-text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from anon_proxy.telemetry_report import render_report


# ---------- Trend panel ----------


def test_report_includes_trend_panel(tmp_path, capsys):
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"date": "2026-04-29", "label_counts": {"EMAIL": 12}, "leak_back": 0, "total_records": 30})
        + "\n"
        + json.dumps({"date": "2026-04-30", "label_counts": {"EMAIL": 3}, "leak_back": 1, "total_records": 28})
        + "\n"
    )
    (tmp_path / "telemetry-raw.jsonl").write_text("")
    render_report(tmp_path)
    out = capsys.readouterr().out
    assert "Trend" in out
    assert "EMAIL" in out
    assert "2026-04-29" in out


# ---------- Leak-back alerts ----------


def test_report_alerts_on_leak_back(tmp_path, capsys):
    (tmp_path / "telemetry-raw.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-04-29T00:00:00Z",
                "schema": 3,
                "spans": [
                    {"label": "EMAIL", "source": "ml", "kept": True, "side": "response"},
                ],
            }
        )
        + "\n"
    )
    (tmp_path / "metrics.jsonl").write_text("")
    render_report(tmp_path)
    out = capsys.readouterr().out
    assert "leak-back" in out.lower()
    assert "1" in out


def test_report_no_leak_back_alert_when_clean(tmp_path, capsys):
    (tmp_path / "telemetry-raw.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-04-29T00:00:00Z",
                "schema": 3,
                "spans": [
                    {"label": "EMAIL", "source": "ml", "kept": True, "side": "user"},
                ],
            }
        )
        + "\n"
    )
    (tmp_path / "metrics.jsonl").write_text("")
    render_report(tmp_path)
    out = capsys.readouterr().out
    # Either no leak-back section or shows 0
    if "leak-back" in out.lower():
        assert "0" in out


# ---------- Storage health ----------


def test_report_storage_health_summary(tmp_path, capsys):
    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_path.write_text(
        json.dumps({"ts": "2026-04-12T00:00:00Z", "schema": 3, "spans": []}) + "\n"
        + json.dumps({"ts": "2026-04-29T00:00:00Z", "schema": 3, "spans": []}) + "\n"
    )
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps(
            {"id": "r1", "ts": "2026-04-15T00:00:00Z", "review": {"label": "EMAIL", "decision": "keep"}}
        )
        + "\n"
    )
    (tmp_path / "metrics.jsonl").write_text("")
    render_report(tmp_path)
    out = capsys.readouterr().out
    assert "Raw:" in out
    assert "2 records" in out
    assert "2026-04-12" in out  # oldest date shown
    assert "Corpus: 1" in out


# ---------- --with-text flag ----------


def test_report_with_text_decrypts_spans(tmp_path, capsys, fake_keyring, monkeypatch):
    from anon_proxy.crypto import generate_key, store_key, encrypt_field

    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    key = generate_key()
    store_key(key)
    (tmp_path / "telemetry-raw.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-04-29T00:00:00Z",
                "schema": 3,
                "spans": [
                    {
                        "label": "EMAIL",
                        "source": "ml",
                        "kept": True,
                        "score": 0.9,
                        "enc_text": encrypt_field("alice@example.com", key),
                    }
                ],
            }
        )
        + "\n"
    )
    (tmp_path / "metrics.jsonl").write_text("")
    render_report(tmp_path, with_text=True)
    captured = capsys.readouterr()
    out = captured.out
    err = captured.err
    assert "alice@example.com" in out
    assert "key resolved" in err.lower()


def test_report_default_does_not_require_key(tmp_path, capsys, fake_keyring, monkeypatch):
    """No keyring entry -> default report still works without prompting / raising."""
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    (tmp_path / "telemetry-raw.jsonl").write_text("")
    (tmp_path / "metrics.jsonl").write_text("")
    # Must not raise KeyNotFoundError
    render_report(tmp_path, with_text=False)


def test_report_default_runs_with_encrypted_records_no_key(tmp_path, capsys, fake_keyring, monkeypatch):
    """Encrypted records present but no key — default report should still work
    (it doesn't decrypt, so no key needed)."""
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    (tmp_path / "telemetry-raw.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-04-29T00:00:00Z",
                "schema": 3,
                "spans": [
                    {
                        "label": "EMAIL",
                        "source": "ml",
                        "kept": True,
                        "score": 0.9,
                        "enc_text": "v1:abc...",
                    }
                ],
            }
        )
        + "\n"
    )
    (tmp_path / "metrics.jsonl").write_text("")
    render_report(tmp_path, with_text=False)  # must not raise
    out = capsys.readouterr().out
    assert "EMAIL" in out  # label is cleartext, shown
