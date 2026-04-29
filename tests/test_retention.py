import json
from pathlib import Path

import pytest

from anon_proxy.retention import RawWriter, CorpusWriter, MetricsWriter, RetentionConfig


def _read_lines(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_raw_writer_appends_records(tmp_path):
    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=30, raw_size_mb=50)
    w = RawWriter(cfg)
    w.write({"ts": "2026-04-29T00:00:00Z", "schema": 3, "spans": []})
    w.write({"ts": "2026-04-29T00:00:01Z", "schema": 3, "spans": []})
    lines = _read_lines(tmp_path / "telemetry-raw.jsonl")
    assert len(lines) == 2


def test_raw_writer_ttl_drops_old_records(tmp_path):
    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=30, raw_size_mb=50)
    w = RawWriter(cfg)
    w.write({"ts": "2026-01-01T00:00:00Z", "schema": 3, "spans": []})  # old
    w.write({"ts": "2026-04-29T00:00:00Z", "schema": 3, "spans": []})  # new
    lines = _read_lines(tmp_path / "telemetry-raw.jsonl")
    assert len(lines) == 1
    assert lines[0]["ts"].startswith("2026-04-29")


def test_raw_writer_size_cap_drops_oldest(tmp_path):
    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=365, raw_size_mb=0)  # 0 MB → drop everything past current
    w = RawWriter(cfg)
    big_payload = "x" * 1024
    for i in range(10):
        w.write({"ts": f"2026-04-29T00:00:{i:02d}Z", "schema": 3, "blob": big_payload})
    lines = _read_lines(tmp_path / "telemetry-raw.jsonl")
    assert len(lines) <= 1


def test_corpus_writer_no_auto_purge(tmp_path):
    w = CorpusWriter(tmp_path)
    w.write({"id": "r1", "ts": "2025-01-01T00:00:00Z", "review": {"label": "EMAIL"}})
    w.write({"id": "r2", "ts": "2026-04-29T00:00:00Z", "review": {"label": "PERSON"}})
    lines = _read_lines(tmp_path / "corpus.jsonl")
    assert len(lines) == 2  # nothing auto-purged


def test_metrics_writer_appends_daily_rollup(tmp_path):
    w = MetricsWriter(tmp_path)
    w.append({"date": "2026-04-29", "label_counts": {"EMAIL": 3}, "leak_back": 0})
    w.append({"date": "2026-04-30", "label_counts": {"EMAIL": 5}, "leak_back": 1})
    lines = _read_lines(tmp_path / "metrics.jsonl")
    assert lines[0]["date"] == "2026-04-29"
    assert lines[1]["date"] == "2026-04-30"


def test_rollup_written_before_drop(tmp_path):
    from anon_proxy.retention import MetricsWriter
    cfg = RetentionConfig(raw_dir=tmp_path, ttl_days=0, raw_size_mb=50)  # everything is "old"
    metrics = MetricsWriter(tmp_path)
    raw = RawWriter(cfg, metrics_writer=metrics)
    raw.write({"ts": "2026-04-29T00:00:00Z", "schema": 3, "spans": [
        {"label": "EMAIL", "source": "ml", "kept": True}
    ]})
    raw.write({"ts": "2026-04-29T00:00:01Z", "schema": 3, "spans": []})
    metrics_lines = _read_lines(tmp_path / "metrics.jsonl")
    assert any(m["label_counts"].get("EMAIL", 0) >= 1 for m in metrics_lines)
