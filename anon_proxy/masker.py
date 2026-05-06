import hashlib
import json
import re
from typing import Callable, Protocol

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


# Patterns for content that should never be masked (non-user PII content)
_SKIP_MASK_PATTERNS = [
    # Claude Code system-reminder blocks - contain tool definitions, skills list, etc.
    re.compile(r"^\s*<system-reminder>", re.MULTILINE),
]


class Masker:
    """Masker runs the user-output pipeline: detect_ml → detect_user → resolve → replace.

    Stages are named methods (`_detect_ml`, `_detect_user`, `_resolve`,
    `_replace`) so each one has a single responsibility and can be tested or
    swapped in isolation.

    `extra_detectors` participate in resolution and affect masked output.

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
        overlap_policy: OverlapPolicy | None = None,
        skip_patterns: list[re.Pattern] | None = None,
        cache_size: int = 256,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._policy: OverlapPolicy = overlap_policy or GreedyLongerWins()
        self._skip_patterns = skip_patterns or _SKIP_MASK_PATTERNS
        self._cache_size = cache_size
        self._cache: dict[str, str] = {}

    @property
    def store(self) -> PIIStore:
        return self._store

    def mask(self, text: str) -> str:
        # Skip-pattern fast path: text we explicitly do not mask.
        for pattern in self._skip_patterns:
            if pattern.search(text):
                return text

        # Cache fast path: identical input → identical output.
        content_hash = _hash_content(text)
        if (cached := self._cache.get(content_hash)) is not None:
            return cached

        ml_spans = self._detect_ml(text)
        user_spans = self._detect_user(text)
        result = self._resolve(ml_spans + user_spans)
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
