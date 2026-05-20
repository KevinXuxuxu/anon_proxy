import contextlib
import contextvars
import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, Protocol

from anon_proxy.mapping import PIIStore, normalize_label
from anon_proxy.privacy_filter import PIIEntity, PrivacyFilter


_TELEMETRY: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "anon_proxy_masker_telemetry", default=None,
)


@contextlib.contextmanager
def telemetry_scope():
    """Collect per-call masker telemetry into a fresh list for the current task.

    Each entry: {"op": "mask"|"unmask"|"unmask_json", "chars": int, "ms": float,
    "cache_hit": bool (mask only), "skipped": bool (mask, optional)}.
    Safe under concurrent asyncio tasks — each task gets its own list via contextvars.
    """
    record: list = []
    token = _TELEMETRY.set(record)
    try:
        yield record
    finally:
        _TELEMETRY.reset(token)


class Detector(Protocol):
    def detect(self, text: str) -> list[PIIEntity]: ...


# Patterns for content that should never be masked (non-user PII content)
_SKIP_MASK_PATTERNS = [
    # Claude Code system-reminder blocks - contain tool definitions, skills list, etc.
    re.compile(r'^\s*<system-reminder>', re.MULTILINE),
    # Tool result blocks that are purely structural (e.g., file listings, tool outputs)
    # These can be extended as needed
]

# Matches placeholder tokens emitted by PIIStore (see mapping.py: f"<{LABEL}_{N}>").
_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")


class Masker:
    """Composes PrivacyFilter + PIIStore to mask outgoing text and unmask LLM replies.

    One Masker instance per conversation: the store accumulates entities across
    turns so the same PII always gets the same placeholder.

    Detection runs in two passes: every detector in `extra_detectors` (typically
    regex) runs first and its matches are substituted inline; then the ML model
    runs on the partially-masked text. Regex hits therefore take precedence over
    the model, and the model still sees full surrounding context (regex matches
    appear as compact `<LABEL_N>` tokens). Within each pass, overlapping spans
    are resolved by preferring the longer span (ties broken by score).

    Performance optimizations:
    - LRU caches detection results by content hash to avoid re-scanning identical text
    - LRU caches masked block-shaped objects by content hash so repeated message
      blocks (the common shape of conversation history) skip re-walking entirely
    - Skips masking for known non-PII patterns (e.g., system-reminders)
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
        extra_detectors: list[Detector] | None = None,
        skip_patterns: list[re.Pattern] | None = None,
        ignore_labels: Iterable[str] | None = None,
        cache_size: int = 4096,
    ) -> None:
        self._filter = filter if filter is not None else PrivacyFilter()
        self._store = store if store is not None else PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._skip_patterns = (
            skip_patterns if skip_patterns is not None else _SKIP_MASK_PATTERNS
        )
        self._ignore_labels: frozenset[str] = frozenset(
            normalize_label(s) for s in (ignore_labels or ())
        )
        self._cache_size = cache_size
        # LRU cache: content_hash -> (entities, masked_text)
        self._cache: OrderedDict[str, tuple[list[PIIEntity], str]] = OrderedDict()
        # LRU cache: block_hash -> already-masked block-shaped object
        self._block_cache: OrderedDict[str, Any] = OrderedDict()

    @property
    def store(self) -> PIIStore:
        return self._store

    def mask(self, text: str) -> str:
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0

        # Empty / whitespace-only input has no PII by definition — skip both
        # passes (and the cache) so the pipeline is never invoked.
        if not text.strip():
            return text

        # Fast path: check if this text matches any skip pattern
        for pattern in self._skip_patterns:
            if pattern.search(text):
                if record is not None:
                    record.append({
                        "op": "mask", "chars": len(text),
                        "ms": (time.perf_counter() - t0) * 1000,
                        "cache_hit": False, "skipped": True,
                    })
                return text  # Skip masking entirely

        # Check cache
        content_hash = _hash_content(text)
        if cached := self._cache.get(content_hash):
            self._cache.move_to_end(content_hash)
            if record is not None:
                record.append({
                    "op": "mask", "chars": len(text),
                    "ms": (time.perf_counter() - t0) * 1000,
                    "cache_hit": True,
                })
            return cached[1]

        # Pass 1: regex detectors first. Substitute matches inline so the ML
        # model sees full context with regex-confirmed PII collapsed to short
        # placeholder tokens — preserves transformer context, prevents the model
        # from second-guessing high-precision regex hits.
        regex_entities: list[PIIEntity] = []
        for detector in self._extra:
            regex_entities.extend(detector.detect(text))
        regex_entities = _resolve_overlaps(regex_entities)
        intermediate = self._substitute(text, regex_entities)

        # Pass 2: ML model on the regex-masked text. Defensively drop any span
        # that intersects a placeholder token; substituting inside one would
        # corrupt the token and break unmask.
        ml_entities = _drop_placeholder_overlaps(
            self._filter.detect(intermediate), intermediate,
        )
        if self._ignore_labels:
            ml_entities = [
                e for e in ml_entities
                if normalize_label(e.label) not in self._ignore_labels
            ]
        ml_entities = _resolve_overlaps(ml_entities)
        masked = self._substitute(intermediate, ml_entities)

        self._cache_result(content_hash, regex_entities + ml_entities, masked)
        if record is not None:
            record.append({
                "op": "mask", "chars": len(text),
                "ms": (time.perf_counter() - t0) * 1000,
                "cache_hit": False,
            })
        return masked

    def _cache_result(self, content_hash: str, entities: list[PIIEntity], masked: str) -> None:
        """Cache a detection result with LRU eviction."""
        self._cache[content_hash] = (entities, masked)
        self._cache.move_to_end(content_hash)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _substitute(self, text: str, entities: list[PIIEntity]) -> str:
        """Replace entities with placeholder tokens.

        Tokens are allocated left-to-right (so the leftmost entity gets the
        lowest index — matches human reading order) but applied right-to-left
        so earlier spans' offsets stay valid as the text is rewritten.
        """
        if not entities:
            return text
        ordered = sorted(entities, key=lambda e: e.start)
        tokens = [self._store.get_or_create(e.label, e.text).token for e in ordered]
        masked = text
        for e, token in zip(reversed(ordered), reversed(tokens)):
            masked = masked[: e.start] + token + masked[e.end :]
        return masked

    def mask_obj(self, obj: Any, walker: Callable[[Any], Any]) -> Any:
        """Mask a JSON-shaped value (typically a message or content block) with
        a content-hash cache so repeated objects skip re-walking entirely.

        `walker` is the function that produces a freshly-masked version of `obj`
        from scratch — invoked only on a cache miss. The returned masked object
        is cached and shared across callers; do not mutate it.

        Designed for conversation history: at turn N+1, `messages[0..N-1]` are
        byte-identical to turn N (and already contain only placeholder tokens),
        so a hash hit short-circuits the entire recursive walk.
        """
        try:
            key = _hash_obj(obj)
        except (TypeError, ValueError):
            return walker(obj)
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0
        cached = self._block_cache.get(key)
        if cached is not None:
            self._block_cache.move_to_end(key)
            if record is not None:
                record.append({
                    "op": "mask_obj",
                    "ms": (time.perf_counter() - t0) * 1000,
                    "cache_hit": True,
                })
            return cached
        result = walker(obj)
        self._block_cache[key] = result
        self._block_cache.move_to_end(key)
        while len(self._block_cache) > self._cache_size:
            self._block_cache.popitem(last=False)
        if record is not None:
            record.append({
                "op": "mask_obj",
                "ms": (time.perf_counter() - t0) * 1000,
                "cache_hit": False,
            })
        return result

    def unmask(self, text: str) -> str:
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0
        result = self._sub(text, lambda s: s)
        if record is not None:
            record.append({
                "op": "unmask", "chars": len(text),
                "ms": (time.perf_counter() - t0) * 1000,
            })
        return result

    def unmask_json(self, text: str) -> str:
        """Unmask tokens sitting inside a JSON string context.

        Replacements are JSON-escaped so an original containing `"`, `\\`, or
        control chars doesn't break the surrounding JSON. Use this for raw
        JSON fragments like Anthropic's `input_json_delta.partial_json` where
        the unmasked text flows through an unparsed string.
        """
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0
        result = self._sub(text, lambda s: json.dumps(s)[1:-1])
        if record is not None:
            record.append({
                "op": "unmask_json", "chars": len(text),
                "ms": (time.perf_counter() - t0) * 1000,
            })
        return result

    def _sub(self, text: str, transform: Callable[[str], str]) -> str:
        """Substitute placeholder tokens with their original values."""
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


def _drop_placeholder_overlaps(entities: list[PIIEntity], text: str) -> list[PIIEntity]:
    """Drop entities whose spans intersect a placeholder token in `text`.

    Touching boundaries (entity.end == placeholder.start or vice versa) are
    allowed, matching the touching-is-not-overlap rule in `_resolve_overlaps`.
    """
    placeholders = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]
    if not placeholders:
        return list(entities)
    return [
        e for e in entities
        if not any(e.start < pe and e.end > ps for ps, pe in placeholders)
    ]


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep a non-overlapping subset of spans (longest-first selection).

    Sort by (-length, -score, start, label) so longer, then higher-confidence,
    then earlier, then alphabetically-labeled spans are considered first. Walk
    the sorted list; keep each candidate iff it does not overlap any already-
    kept span. Touching at boundaries (`prev.end == next.start`) does not count
    as overlap. Return kept spans sorted by start position so callers can
    substitute right-to-left.

    This replaces the previous greedy-by-start algorithm, which could drop a
    span that did not actually conflict with the eventual winner (e.g. given
    A=[0,5], B=[4,10], C=[7,15] the old algorithm chained A→B→C and lost A,
    even though A and C do not overlap).
    """
    if not entities:
        return []
    candidates = sorted(
        entities,
        key=lambda e: (-(e.end - e.start), -e.score, e.start, e.label),
    )
    kept: list[PIIEntity] = []
    for e in candidates:
        if any(e.start < k.end and e.end > k.start for k in kept):
            continue
        kept.append(e)
    return sorted(kept, key=lambda e: e.start)


def _hash_content(text: str) -> str:
    """Hash content for caching detection results.

    Uses SHA256 truncated to 12 chars (collision-resistant enough for cache keys,
    compact enough to be memory-efficient).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _hash_obj(obj: Any) -> str:
    """Hash a JSON-shaped value by its serialized form.

    Insertion order is preserved end-to-end (json.loads → dict copy → json.dumps),
    so we deliberately skip sort_keys: it's a 5–10x slowdown on large nested
    objects and adds no value for our pipeline.
    """
    serialized = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
