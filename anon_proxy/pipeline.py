"""Named PII pipeline stages: detect → resolve → replace.

Currently a thin layer over what `Masker` and `_resolve_overlaps` already do —
the point of this module is to give every stage a name and an interface, so
the maintainer's three questions (passes, range interaction, chunking) have
named code to point at.

`AttributedSpan` tags every detected span with the detector that emitted it.
`OverlapPolicy` is the seam where range interaction lives.
`OverlapEvent` is the data shape the telemetry layer reads to record kept-vs-
dropped decisions, without scattering instrumentation through the resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from anon_proxy.privacy_filter import PIIEntity

Source = Literal["ml", "user_regex", "baseline"]
_VALID_SOURCES: frozenset[str] = frozenset({"ml", "user_regex", "baseline"})

Reason = Literal["overlap_longer", "overlap_score_tie"]


@dataclass(frozen=True)
class AttributedSpan:
    """A detected span plus the detector that produced it."""

    entity: PIIEntity
    source: Source

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"AttributedSpan.source must be one of {sorted(_VALID_SOURCES)}, got {self.source!r}"
            )

    @property
    def start(self) -> int:
        return self.entity.start

    @property
    def end(self) -> int:
        return self.entity.end

    @property
    def label(self) -> str:
        return self.entity.label

    @property
    def length(self) -> int:
        return self.entity.end - self.entity.start


@dataclass(frozen=True)
class OverlapEvent:
    """Records that `winner` beat `loser` during overlap resolution."""

    winner: AttributedSpan
    loser: AttributedSpan
    reason: Reason


@dataclass(frozen=True)
class ResolveResult:
    """Output of an `OverlapPolicy.resolve` call."""

    kept: list[AttributedSpan]
    events: list[OverlapEvent]


class OverlapPolicy(Protocol):
    def resolve(self, spans: list[AttributedSpan]) -> ResolveResult: ...
