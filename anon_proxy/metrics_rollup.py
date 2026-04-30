"""Daily aggregate rollup. Pure transformation: takes records, produces / updates DailyRollup."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DailyRollup:
    date: str
    label_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    leak_back: int = 0
    total_records: int = 0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "label_counts": self.label_counts,
            "source_counts": self.source_counts,
            "leak_back": self.leak_back,
            "total_records": self.total_records,
        }


def update_daily_rollup(state: dict[str, DailyRollup], record: dict) -> None:
    date = record["ts"][:10]
    rollup = state.setdefault(date, DailyRollup(date=date))
    rollup.total_records += 1
    for span in record.get("spans", []):
        label = span["label"]
        rollup.label_counts[label] = rollup.label_counts.get(label, 0) + 1
        src = span.get("source", "?")
        rollup.source_counts[src] = rollup.source_counts.get(src, 0) + 1
        if span.get("side") == "response":
            rollup.leak_back += 1
