import json
import re
from typing import Callable, Protocol

from anon_proxy.mapping import PIIStore
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


class Masker:
    """Composes PrivacyFilter + PIIStore to mask outgoing text and unmask LLM replies.

    One Masker instance per conversation: the store accumulates entities across
    turns so the same PII always gets the same placeholder.

    `extra_detectors` is a list of objects with a `detect(text) -> list[PIIEntity]`
    method whose spans are merged into the primary filter's output. Overlapping
    spans from different detectors are resolved by preferring the longer span.

    `telemetry` is an optional observer that sees the raw ML output and the
    user's configured regex output for each mask() call. It never affects
    masking behavior — see `anon_proxy.telemetry`.
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
        extra_detectors: list[Detector] | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._telemetry = telemetry

    @property
    def store(self) -> PIIStore:
        return self._store

    def mask(self, text: str) -> str:
        ml_entities: list[PIIEntity] = list(self._filter.detect(text))
        extra_entities: list[PIIEntity] = []
        for detector in self._extra:
            extra_entities.extend(detector.detect(text))
        if self._telemetry is not None:
            self._telemetry.observe(text, ml_entities, extra_entities)
        entities = _resolve_overlaps(ml_entities + extra_entities)
        # Replace right-to-left so earlier spans' offsets stay valid.
        for e in sorted(entities, key=lambda x: x.start, reverse=True):
            token = self._store.get_or_create(e.label, e.text).token
            text = text[: e.start] + token + text[e.end :]
        return text

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
