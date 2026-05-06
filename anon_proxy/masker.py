import hashlib
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


# Patterns for content that should never be masked (non-user PII content)
_SKIP_MASK_PATTERNS = [
    # Claude Code system-reminder blocks - contain tool definitions, skills list, etc.
    re.compile(r"^\s*<system-reminder>", re.MULTILINE),
]


class Masker:
    """Masker runs the user-output pipeline: detect_ml → detect_user → resolve → replace.

    Stages are named methods (`_detect_ml`, `_detect_user`, `_resolve`,
    `_replace`) so each one has a single responsibility and can be tested or
    swapped in isolation. `_observe` is a thin dispatch into the optional
    telemetry sink and never touches the masked output.

    `extra_detectors` participate in resolution and affect masked output.
    A baseline regex set, when present, lives in the telemetry observer
    (`anon_proxy.telemetry.TelemetryObserver`) — observer-only, never affects
    Masker output.

    Performance optimizations:
    - Caches mask results by content hash to skip re-scanning identical text
    - Skips masking for known non-PII patterns (e.g., system-reminders)
    - Early-return on empty resolve result
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
        extra_detectors: list[Detector] | None = None,
        telemetry: TelemetrySink | None = None,
        overlap_policy: OverlapPolicy | None = None,
        skip_patterns: list[re.Pattern] | None = None,
        cache_size: int = 256,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._telemetry = telemetry
        self._policy: OverlapPolicy = overlap_policy or GreedyLongerWins()
        self._skip_patterns = skip_patterns or _SKIP_MASK_PATTERNS
        self._cache_size = cache_size
        self._cache: dict[str, str] = {}

    @property
    def store(self) -> PIIStore:
        return self._store

    @property
    def telemetry(self) -> TelemetrySink | None:
        return self._telemetry

    def detect_only(self, text: str) -> tuple[list[AttributedSpan], list[AttributedSpan]]:
        """Return (ml_spans, user_spans) without resolving overlaps or replacing.

        Used by adapters for response-side telemetry where we want detector
        signal but do not want to mutate text or touch the PII store.
        """
        return self._detect_ml(text), self._detect_user(text)

    def mask(self, text: str) -> str:
        # Skip-pattern fast path: text we explicitly do not mask.
        for pattern in self._skip_patterns:
            if pattern.search(text):
                return text

        # Cache fast path: identical input → identical output. Telemetry still
        # fires below — the cache only short-circuits the actual replace work.
        content_hash = _hash_content(text)
        cached = self._cache.get(content_hash)

        ml_spans = self._detect_ml(text)
        user_spans = self._detect_user(text)
        result = self._resolve(ml_spans + user_spans)
        if self._telemetry is not None:
            sink = _current_batch.get() or self._telemetry
            self._observe(sink, text, ml_spans, user_spans, result)

        if cached is not None:
            return cached
        if not result.kept:
            self._cache_result(content_hash, text)
            return text
        masked = self._replace(text, result.kept)
        self._cache_result(content_hash, masked)
        return masked

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

    def _cache_result(self, content_hash: str, masked: str) -> None:
        """Cache a mask result, evicting oldest if cache is full (FIFO)."""
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[content_hash] = masked

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

        Reentrant: a nested scope from the same Masker reuses the outer batch and
        does NOT commit when it exits — only the outermost scope commits.
        If a different Masker enters a scope, it creates a new batch.
        """
        if self._telemetry is None:
            yield None
            return
        existing = _current_batch.get()
        if existing is not None and getattr(existing, "_owner", None) is self:
            yield existing
            return
        batch = self._telemetry.new_batch()
        batch._owner = self
        token = _current_batch.set(batch)
        try:
            yield batch
        finally:
            # reset() may raise ValueError when the streaming body_iter()
            # generator resumes in a different async Context (the token was
            # created in _handle_messages' context, but body_iter's finally
            # runs in the generator's context). Swallow it — the important
            # thing is that commit() always runs.
            try:
                _current_batch.reset(token)
            except ValueError:
                pass
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


def _hash_content(text: str) -> str:
    """Hash content for caching detection results.

    SHA256 truncated to 12 chars: collision-resistant enough for cache keys,
    compact enough to be memory-efficient.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
