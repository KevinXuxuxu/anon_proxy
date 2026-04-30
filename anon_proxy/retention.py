"""Three-tier writer layer with TTL + size auto-purge for raw, indefinite for corpus/metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from anon_proxy.storage_paths import secure_create_dir, secure_create_file


@dataclass(frozen=True)
class RetentionConfig:
    raw_dir: Path
    ttl_days: int = 30
    raw_size_mb: int = 50

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / "telemetry-raw.jsonl"

    @property
    def corpus_path(self) -> Path:
        return self.raw_dir / "corpus.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.raw_dir / "metrics.jsonl"


class _AppendOnlyWriter:
    def __init__(self, path: Path) -> None:
        secure_create_file(path)
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def _append(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)


class RawWriter(_AppendOnlyWriter):
    """Auto-purges by TTL and size cap on every write."""

    def __init__(self, cfg: RetentionConfig, metrics_writer: "MetricsWriter | None" = None) -> None:
        secure_create_dir(cfg.raw_dir)
        super().__init__(cfg.raw_path)
        self._cfg = cfg
        self._metrics_writer = metrics_writer

    def write(self, record: dict) -> None:
        if "id" not in record:
            record = {**record, "id": uuid4().hex[:12]}
        self._append(record)
        self._purge()

    def _purge(self) -> None:
        max_bytes = self._cfg.raw_size_mb * 1024 * 1024
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._cfg.ttl_days)
        lines = [l for l in self._path.read_text().splitlines() if l.strip()]
        keep_after_ttl = [l for l in lines if _ts(l) >= cutoff]

        # If under size cap after TTL, just write back what survives TTL
        encoded = "\n".join(keep_after_ttl) + ("\n" if keep_after_ttl else "")
        if len(encoded.encode("utf-8")) <= max_bytes:
            if len(keep_after_ttl) < len(lines):
                self._record_dropped(set(lines) - set(keep_after_ttl))
                self._path.write_text(encoded)
            return

        # Over size cap: drop oldest among TTL-survivors until under cap
        keep = list(keep_after_ttl)
        encoded = "\n".join(keep) + ("\n" if keep else "")
        while len(encoded.encode("utf-8")) > max_bytes and keep:
            keep.pop(0)
            encoded = "\n".join(keep) + ("\n" if keep else "")
        dropped = set(lines) - set(keep)
        self._record_dropped(dropped)
        self._path.write_text(encoded)

    def _record_dropped(self, dropped_lines: set[str]) -> None:
        """Aggregate dropped lines into daily rollups and persist before they're lost."""
        if not self._metrics_writer or not dropped_lines:
            return
        from anon_proxy.metrics_rollup import update_daily_rollup
        state: dict = {}
        for line in dropped_lines:
            try:
                update_daily_rollup(state, json.loads(line))
            except Exception:
                continue
        for date, rollup in sorted(state.items()):
            self._metrics_writer.append(rollup.to_dict())


class CorpusWriter(_AppendOnlyWriter):
    """No auto-purge. Manual via `anon-proxy telemetry purge --corpus <id>`."""

    def __init__(self, root_dir: Path) -> None:
        secure_create_dir(root_dir)
        super().__init__(root_dir / "corpus.jsonl")

    def write(self, record: dict) -> None:
        self._append(record)


class MetricsWriter(_AppendOnlyWriter):
    """No auto-purge. Append-only daily rollups."""

    def __init__(self, root_dir: Path) -> None:
        secure_create_dir(root_dir)
        super().__init__(root_dir / "metrics.jsonl")

    def append(self, record: dict) -> None:
        self._append(record)


def _ts(line: str) -> datetime:
    try:
        rec = json.loads(line)
        ts_str = rec["ts"].rstrip("Z")
        return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
    except Exception:
        # Conservatively keep unparseable lines (treat as "very recent")
        return datetime.now(timezone.utc)
