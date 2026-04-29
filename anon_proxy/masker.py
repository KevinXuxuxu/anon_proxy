import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Protocol

from anon_proxy.mapping import PIIStore
from anon_proxy.pipeline import (
    AttributedSpan,
    GreedyLongerWins,
    OverlapPolicy,
    ResolveResult,
)
from anon_proxy.privacy_filter import PIIEntity, PrivacyFilter


class Detector(Protocol):
    def detect(self, text: str) -> list[PIIEntity]: ...


class TelemetrySink(Protocol):
    def observe(
        self,
        text: str,
        ml_entities: list[PIIEntity],
        extra_entities: list[PIIEntity],
    ) -> None: ...

    def new_batch(self) -> "TelemetrySink": ...


# Per-request telemetry batch. ContextVar so async tasks don't cross-contaminate.
_current_batch: ContextVar[object | None] = ContextVar(
    "anon_proxy_telemetry_batch", default=None
)


class Masker:
    """Masker runs the user-output pipeline: detect_ml → detect_user → resolve → replace.

    Stages are named methods (`_detect_ml`, `_detect_user`, `_resolve`,
    `_replace`) so each one has a single responsibility and can be tested or
    swapped in isolation. `_observe` is a thin dispatch into the optional
    telemetry sink and never touches the masked output.

    `extra_detectors` run as Pass 2 of the architecture diagram (their spans
    participate in resolution and affect masked output). The baseline regex
    set referenced as Pass 3 lives in the telemetry observer
    (`anon_proxy.telemetry.TelemetryObserver`) — observer-only, never affects
    Masker output.
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
        extra_detectors: list[Detector] | None = None,
        telemetry: TelemetrySink | None = None,
        overlap_policy: OverlapPolicy | None = None,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._telemetry = telemetry
        self._policy: OverlapPolicy = overlap_policy or GreedyLongerWins()

    @property
    def store(self) -> PIIStore:
        return self._store

    @property
    def telemetry(self) -> TelemetrySink | None:
        return self._telemetry

    def mask(self, text: str) -> str:
        ml_spans = self._detect_ml(text)
        user_spans = self._detect_user(text)
        result = self._resolve(ml_spans + user_spans)
        if self._telemetry is not None:
            sink = _current_batch.get() or self._telemetry
            self._observe(sink, text, ml_spans, user_spans, result)
        return self._replace(text, result.kept)

    def _detect_ml(self, text: str) -> list[AttributedSpan]:
        return [AttributedSpan(entity=e, source="ml") for e in self._filter.detect(text)]

    def _detect_user(self, text: str) -> list[AttributedSpan]:
        out: list[AttributedSpan] = []
        for d in self._extra:
            out.extend(AttributedSpan(entity=e, source="user_regex") for e in d.detect(text))
        return out

    def _resolve(self, spans: list[AttributedSpan]) -> ResolveResult:
        return self._policy.resolve(spans)

    def _replace(self, text: str, kept: list[AttributedSpan]) -> str:
        # Right-to-left so earlier offsets stay valid.
        for s in sorted(kept, key=lambda x: x.start, reverse=True):
            token = self._store.get_or_create(s.label, s.entity.text).token
            text = text[: s.start] + token + text[s.end :]
        return text

    def _observe(
        self,
        sink: object,
        text: str,
        ml_spans: list[AttributedSpan],
        user_spans: list[AttributedSpan],
        result: ResolveResult,
    ) -> None:
        # Backwards-compatible: call observe(text, ml_entities, extra_entities)
        # if sink doesn't support the v2 attributed signature.
        observe_v2 = getattr(sink, "observe_v2", None)
        if callable(observe_v2):
            observe_v2(
                text=text,
                ml_spans=ml_spans,
                user_spans=user_spans,
                kept=result.kept,
                events=result.events,
            )
            return
        sink.observe(
            text,
            [s.entity for s in ml_spans],
            [s.entity for s in user_spans],
        )

    @contextmanager
    def request_scope(self) -> Iterator["object | None"]:
        """Aggregate all mask() calls inside this scope into one telemetry record.

        Yields the active `TelemetryBatch` (or None when telemetry is disabled)
        so callers can record extra fields (e.g. latency) before the batch
        commits at scope exit.

        Reentrant: a nested scope reuses the outer batch and does NOT commit
        when it exits — only the outermost scope commits.
        """
        if self._telemetry is None:
            yield None
            return
        existing = _current_batch.get()
        if existing is not None:
            yield existing
            return
        batch = self._telemetry.new_batch()
        token = _current_batch.set(batch)
        try:
            yield batch
        finally:
            _current_batch.reset(token)
            commit = getattr(batch, "commit", None)
            if callable(commit):
                commit()

    def unmask(self, text: str) -> str:
        return self._sub(text, lambda s: s)

    def unmask_json(self, text: str) -> str:
        """Unmask tokens inside a JSON string context (escapes the unmasked value)."""
        return self._sub(text, lambda s: json.dumps(s)[1:-1])

    def _sub(self, text: str, transform: Callable[[str], str]) -> str:
        tokens = self._store.tokens()
        if not tokens:
            return text
        # Longest-first so "<PERSON_1>" can't shadow "<PERSON_10>".
        pattern = re.compile(
            "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
        )

        def repl(m: re.Match[str]) -> str:
            original = self._store.original(m.group(0))
            return transform(original) if original is not None else m.group(0)

        return pattern.sub(repl, text)
