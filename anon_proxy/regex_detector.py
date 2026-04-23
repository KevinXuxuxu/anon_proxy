"""User-defined regex-based PII detection.

Complements the ML detector for kinds of PII it doesn't reliably catch
(IP addresses, SSNs, credit cards, internal IDs) or that a specific deployment
wants handled deterministically.

Config file is a flat JSON object mapping label -> regex:

    {
      "SSN":  "\\b\\d{3}-\\d{2}-\\d{4}\\b",
      "IPV4": "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b"
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from anon_proxy.privacy_filter import PIIEntity


class RegexDetector:
    """Emits PIIEntity spans for every match of each configured pattern."""

    def __init__(self, patterns: dict[str, str]) -> None:
        compiled: list[tuple[str, re.Pattern[str]]] = []
        errors: list[str] = []
        for label, pattern in patterns.items():
            try:
                compiled.append((label, re.compile(pattern)))
            except re.error as e:
                errors.append(f"  {label!r}: {e}")
        if errors:
            raise ValueError("invalid regex patterns:\n" + "\n".join(errors))
        self._patterns = compiled

    def detect(self, text: str) -> list[PIIEntity]:
        out: list[PIIEntity] = []
        for label, rx in self._patterns:
            for m in rx.finditer(text):
                start, end = m.span()
                if start == end:
                    continue
                out.append(
                    PIIEntity(
                        label=label,
                        text=text[start:end],
                        start=start,
                        end=end,
                        score=1.0,
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self._patterns)


def load_patterns(path: str | Path) -> dict[str, str]:
    """Parse and validate a patterns JSON file. Raises ValueError on bad shape."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON — {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object mapping label -> regex")
    bad = [k for k, v in data.items() if not (isinstance(k, str) and isinstance(v, str))]
    if bad:
        raise ValueError(f"{path}: non-string label or pattern for keys: {bad!r}")
    return data
