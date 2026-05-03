import hashlib
import json
import re
from typing import Callable, Protocol

from anon_proxy.mapping import PIIStore
from anon_proxy.privacy_filter import PIIEntity, PrivacyFilter


class Detector(Protocol):
    def detect(self, text: str) -> list[PIIEntity]: ...


# Patterns for content that should never be masked (non-user PII content)
_SKIP_MASK_PATTERNS = [
    # Claude Code system-reminder blocks - contain tool definitions, skills list, etc.
    re.compile(r'^\s*<system-reminder>', re.MULTILINE),
    # Tool result blocks that are purely structural (e.g., file listings, tool outputs)
    # These can be extended as needed
]


class Masker:
    """Composes PrivacyFilter + PIIStore to mask outgoing text and unmask LLM replies.

    One Masker instance per conversation: the store accumulates entities across
    turns so the same PII always gets the same placeholder.

    `extra_detectors` is a list of objects with a `detect(text) -> list[PIIEntity]`
    method whose spans are merged into the primary filter's output. Overlapping
    spans from different detectors are resolved by preferring the longer span.

    Performance optimizations:
    - Caches detection results by content hash to avoid re-scanning identical text
    - Skips masking for known non-PII patterns (e.g., system-reminders)
    - Early-return if content already contains only placeholders (no new PII)
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
        extra_detectors: list[Detector] | None = None,
        skip_patterns: list[re.Pattern] | None = None,
        cache_size: int = 256,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._skip_patterns = skip_patterns or _SKIP_MASK_PATTERNS
        self._cache_size = cache_size
        # Cache: content_hash -> (entities_text, masked_text)
        self._cache: dict[str, tuple[list[PIIEntity], str]] = {}

    @property
    def store(self) -> PIIStore:
        return self._store

    def mask(self, text: str) -> str:
        # Fast path: check if this text matches any skip pattern
        for pattern in self._skip_patterns:
            if pattern.search(text):
                return text  # Skip masking entirely

        # Check cache
        content_hash = _hash_content(text)
        if cached := self._cache.get(content_hash):
            return cached[1]

        # Detect entities
        entities: list[PIIEntity] = list(self._filter.detect(text))
        for detector in self._extra:
            entities.extend(detector.detect(text))
        entities = _resolve_overlaps(entities)

        # Early return if no entities found
        if not entities:
            self._cache_result(content_hash, [], text)
            return text

        # Replace right-to-left so earlier spans' offsets stay valid.
        masked = text
        for e in sorted(entities, key=lambda x: x.start, reverse=True):
            token = self._store.get_or_create(e.label, e.text).token
            masked = masked[: e.start] + token + masked[e.end :]

        self._cache_result(content_hash, entities, masked)
        return masked

    def _cache_result(self, content_hash: str, entities: list[PIIEntity], masked: str) -> None:
        """Cache a detection result, evicting oldest if cache is full."""
        if len(self._cache) >= self._cache_size:
            # Simple FIFO eviction - remove first item
            self._cache.pop(next(iter(self._cache)))
        self._cache[content_hash] = (entities, masked)

    def unmask(self, text: str) -> str:
        return self._sub(text, lambda s: s)

    def unmask_json(self, text: str) -> str:
        """Unmask tokens sitting inside a JSON string context.

        Replacements are JSON-escaped so an original containing `"`, `\\`, or
        control chars doesn't break the surrounding JSON. Use this for raw
        JSON fragments like Anthropic's `input_json_delta.partial_json` where
        the unmasked text flows through an unparsed string.
        """
        return self._sub(text, lambda s: json.dumps(s)[1:-1])

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


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep a non-overlapping subset of spans.

    Greedy: sort by (start, -length, -score) so earlier and longer spans land first.
    Walk left-to-right; when a span overlaps the last kept, replace only if the
    new one is strictly longer (ties: higher score wins). Touching spans at
    `prev.end == next.start` do not overlap.
    """
    if not entities:
        return entities
    ordered = sorted(
        entities,
        key=lambda e: (e.start, -(e.end - e.start), -e.score, e.label),
    )
    kept: list[PIIEntity] = []
    for e in ordered:
        if kept and e.start < kept[-1].end:
            prev = kept[-1]
            prev_len = prev.end - prev.start
            cur_len = e.end - e.start
            if cur_len > prev_len or (cur_len == prev_len and e.score > prev.score):
                kept[-1] = e
            continue
        kept.append(e)
    return kept


def _hash_content(text: str) -> str:
    """Hash content for caching detection results.

    Uses SHA256 truncated to 12 chars (collision-resistant enough for cache keys,
    compact enough to be memory-efficient).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
