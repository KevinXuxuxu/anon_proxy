"""Tests for anon-proxy telemetry CLI (Phase 8A)."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

from anon_proxy.triage_cli import main


# ---------------------------------------------------------------------------
# 8.1: Dispatcher tests
# ---------------------------------------------------------------------------


def test_missing_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code != 0


def test_subcommand_dispatch_recognized_triage_help():
    with pytest.raises(SystemExit) as e:
        main(["triage", "--help"])
    assert e.value.code == 0


def test_all_documented_subcommands_parse():
    """All subcommands listed in the plan must at least parse without error."""
    for sub in ["triage", "label", "corpus", "purge", "metrics", "export-key", "import-key", "suggest-regex"]:
        with pytest.raises(SystemExit) as e:
            main([sub, "--help"])
        assert e.value.code == 0, f"{sub} --help failed"


# ---------------------------------------------------------------------------
# 8.2: metrics subcommand
# ---------------------------------------------------------------------------


def test_metrics_subcommand_reads_metrics_jsonl(tmp_path, monkeypatch, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"date":"2026-04-29","label_counts":{"EMAIL":3},"leak_back":0,"total_records":5}\n')
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["metrics", "--json"])
    out = capsys.readouterr().out
    assert "EMAIL" in out
    assert "2026-04-29" in out


def test_metrics_subcommand_human_readable(tmp_path, monkeypatch, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"date":"2026-04-29","label_counts":{"EMAIL":12,"PERSON":5},"leak_back":0,"total_records":30}\n'
        '{"date":"2026-04-30","label_counts":{"EMAIL":3},"leak_back":1,"total_records":28}\n'
    )
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["metrics"])  # no --json
    out = capsys.readouterr().out
    assert "2026-04-29" in out
    assert "2026-04-30" in out
    assert "EMAIL" in out


def test_metrics_subcommand_no_file_friendly_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["metrics"])
    err = capsys.readouterr().err
    assert "no metrics" in err.lower() or "not yet" in err.lower()


def test_metrics_subcommand_filters_by_since(tmp_path, monkeypatch, capsys):
    (tmp_path / "metrics.jsonl").write_text(
        '{"date":"2026-04-29","label_counts":{"EMAIL":1},"leak_back":0,"total_records":1}\n'
        '{"date":"2026-04-30","label_counts":{"EMAIL":1},"leak_back":0,"total_records":1}\n'
    )
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["metrics", "--since", "2026-04-30", "--json"])
    out = capsys.readouterr().out
    assert "2026-04-30" in out
    assert "2026-04-29" not in out


# ---------------------------------------------------------------------------
# 8.3: triage --json [--metadata-only]
# ---------------------------------------------------------------------------


def test_triage_json_metadata_only_omits_encrypted_fields(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_path.write_text(json.dumps({
        "ts": "2026-04-29T00:00:00Z",
        "schema": 3,
        "spans": [{
            "label": "EMAIL", "source": "ml", "kept": True, "score": 0.9,
            "enc_text": "v1:abc...", "enc_window": "v1:def...",
        }],
        "enc_text": "v1:full...",
    }) + "\n")
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["triage", "--json", "--metadata-only", "--days", "365"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed[0]["spans"][0]["label"] == "EMAIL"
    assert "enc_text" not in parsed[0]["spans"][0]
    assert "enc_window" not in parsed[0]["spans"][0]
    # Strict: --metadata-only strips top-level enc_text too
    assert "enc_text" not in parsed[0] or parsed[0]["enc_text"] is None


def test_triage_json_with_signatures_decrypts_and_strips_text(tmp_path, monkeypatch, capsys, fake_keyring):
    """Non-metadata-only mode decrypts to compute signatures, then strips raw text."""
    from anon_proxy.crypto import generate_key, store_key, encrypt_field
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    key = generate_key()
    store_key(key)
    raw_path = tmp_path / "telemetry-raw.jsonl"
    # Use gmail.com — a known free-email domain per signatures._FREE_DOMAINS
    raw_path.write_text(json.dumps({
        "ts": "2026-04-29T00:00:00Z",
        "schema": 3,
        "spans": [{
            "label": "EMAIL", "source": "ml", "kept": True, "score": 0.9,
            "enc_text": encrypt_field("alice@gmail.com", key),
            "enc_window": encrypt_field("Email me at alice@gmail.com today", key),
        }],
    }) + "\n")
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["triage", "--json", "--days", "365"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    span = parsed[0]["spans"][0]
    assert span["label"] == "EMAIL"
    assert "signature" in span
    assert span["signature"]["domain_class"] == "free"
    # Encrypted fields stripped from output
    assert "enc_text" not in span
    assert "enc_window" not in span


def test_triage_json_filters_by_source_leak_back(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_path.write_text(
        json.dumps({"ts": "2026-04-29T00:00:00Z", "schema": 3, "spans": [
            {"label": "EMAIL", "source": "ml", "kept": True, "side": "user"}
        ]}) + "\n" +
        json.dumps({"ts": "2026-04-29T00:00:01Z", "schema": 3, "spans": [
            {"label": "EMAIL", "source": "ml", "kept": True, "side": "response"}
        ]}) + "\n"
    )
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    main(["triage", "--json", "--metadata-only", "--source", "leak-back", "--days", "365"])
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed) == 1
    assert parsed[0]["ts"].endswith("00:01Z")


# ---------------------------------------------------------------------------
# 8.4: Interactive triage UI
# ---------------------------------------------------------------------------


def test_interactive_triage_keep_promotes_to_corpus(tmp_path, monkeypatch, fake_keyring):
    from anon_proxy.crypto import generate_key, store_key, encrypt_field
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    key = generate_key()
    store_key(key)
    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_path.write_text(json.dumps({
        "ts": "2026-04-29T00:00:00Z", "schema": 3,
        "spans": [{
            "label": "PERSON", "source": "baseline", "kept": False, "score": 0.0,
            "enc_text": encrypt_field("Project Aurora", key),
            "enc_window": encrypt_field("...about Project Aurora today...", key),
        }],
    }) + "\n")
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("k\nNOT_PII\nq\n"))
    main(["triage", "--days", "365"])
    corpus_path = tmp_path / "corpus.jsonl"
    assert corpus_path.exists()
    corpus_lines = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
    assert len(corpus_lines) == 1
    assert corpus_lines[0]["review"]["decision"] == "keep"
    assert corpus_lines[0]["review"]["label"] == "NOT_PII"


def test_interactive_triage_skip_does_not_write_corpus(tmp_path, monkeypatch, fake_keyring):
    from anon_proxy.crypto import generate_key, store_key, encrypt_field
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    key = generate_key()
    store_key(key)
    raw_path = tmp_path / "telemetry-raw.jsonl"
    raw_path.write_text(json.dumps({
        "ts": "2026-04-29T00:00:00Z", "schema": 3,
        "spans": [{
            "label": "EMAIL", "source": "ml", "kept": True, "score": 0.9,
            "enc_text": encrypt_field("alice@example.com", key),
            "enc_window": encrypt_field("Email me at alice@example.com", key),
        }],
    }) + "\n")
    monkeypatch.setenv("ANON_PROXY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("s\nq\n"))
    main(["triage", "--days", "365"])
    corpus_path = tmp_path / "corpus.jsonl"
    # File may not exist OR may exist but be empty
    if corpus_path.exists():
        assert corpus_path.read_text().strip() == ""
