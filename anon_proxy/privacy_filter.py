import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import pipeline


@dataclass(frozen=True)
class PIIEntity:
    label: str
    text: str
    start: int
    end: int
    score: float


# Per-label characters allowed inside the gap between two adjacent same-label
# spans when deciding whether to merge them. Labels absent from this map can
# only merge across an empty gap.
DEFAULT_MERGE_GAP_ALLOWED: dict[str, str] = {
    # Names: space for "Alice Smith", hyphen for "Jean-Luc", apostrophe for
    # "O'Neil", period for "J.R.".
    "PERSON":       " \t\n-'.",
    # Addresses: whitespace plus the punctuation that commonly splits a
    # single postal address — "123 Main St., Apt #4", "Suite 3B-1".
    "ADDRESS":      " \t\n,.#-/",
    # Dates: "Jan 1, 2025", "2025-01-01", "1/1/25".
    "DATE":         " \t\n,/.-",
    "ORGANIZATION": " \t\n&.,-",
    "LOCATION":     " \t\n,.-",
}


class PrivacyFilter:
    """Thin wrapper around the openai/privacy-filter token classifier.

    The HF pipeline's aggregation_strategy only merges subword pieces within
    a single word. For PII masking we also want to merge *adjacent* same-label
    spans ("Alice" + "Smith" → "Alice Smith", "alice@example" + ".com" →
    "alice@example.com") so each entity maps to one placeholder downstream.
    That second merge pass is `merge_adjacent`, on by default.

    The merge rule is per-label: two same-label spans merge iff either the
    gap between them is empty, or every character of the gap appears in that
    label's allowed-char set. Defaults live in `DEFAULT_MERGE_GAP_ALLOWED`
    (whitespace is NOT automatically mergeable — it has to be in the label's
    entry). Pass `merge_gap_allowed` to override per-label entries; labels
    you don't mention keep their default, and labels you set to `""` can only
    merge across an empty gap.

    Long texts are split into overlapping-free chunks of at most `chunk_size`
    characters (default 1500, ~375 English tokens — safely within BERT's 512
    token limit). Splits happen at the last whitespace before the boundary so
    words are never bisected. Entity spans from adjacent chunks are combined
    before the adjacency-merge pass, so entities that straddle a chunk boundary
    are still collapsed into a single placeholder.
    """

    MODEL_ID = "openai/privacy-filter"

    def __init__(
        self,
        *,
        aggregation_strategy: str = "simple",
        merge_adjacent: bool = True,
        merge_gap_allowed: dict[str, str] | None = None,
        chunk_size: int = 1500,
        device: int | str | None = None,
    ) -> None:
        self._pipe = pipeline(
            task="token-classification",
            model=self.MODEL_ID,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )
        self._merge_adjacent = merge_adjacent
        merged_policy = {**DEFAULT_MERGE_GAP_ALLOWED, **(merge_gap_allowed or {})}
        self._gap_allowed: dict[str, frozenset[str]] = {
            label: frozenset(chars) for label, chars in merged_policy.items()
        }
        self._chunk_size = chunk_size

    def detect(self, text: str) -> list[PIIEntity]:
        chunks = _split_chunks(text, self._chunk_size)
        entities: list[PIIEntity] = []
        for offset, chunk in chunks:
            for r in self._pipe(chunk):
                e = _to_entity(r, chunk)
                entities.append(PIIEntity(
                    label=e.label,
                    text=e.text,
                    start=e.start + offset,
                    end=e.end + offset,
                    score=e.score,
                ))
        if self._merge_adjacent:
            entities = _merge_adjacent_entities(entities, text, self._gap_allowed)
        return entities

    def detect_raw(self, text: str) -> list[dict]:
        """Return the pipeline's untouched per-span dicts for debugging."""
        return list(self._pipe(text))

    def detect_batch(self, texts: Iterable[str]) -> list[list[PIIEntity]]:
        texts = list(texts)
        if not texts:
            return []
        # If any text exceeds the chunk limit, fall back to per-text detect()
        # so chunking is applied correctly for each one.
        if any(len(t) > self._chunk_size for t in texts):
            return [self.detect(t) for t in texts]
        results = self._pipe(texts)
        out: list[list[PIIEntity]] = []
        for t, res in zip(texts, results):
            entities = [_to_entity(r, t) for r in res]
            if self._merge_adjacent:
                entities = _merge_adjacent_entities(entities, t, self._gap_allowed)
            out.append(entities)
        return out


def _split_chunks(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Return (start_offset, chunk_text) pairs covering all of `text`.

    Each chunk is at most `max_chars` characters. Splits at the last
    whitespace before the limit so words are not bisected; falls back to a
    hard cut if no whitespace is found in the window.
    """
    if len(text) <= max_chars:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        if start + max_chars >= len(text):
            chunks.append((start, text[start:]))
            break
        split = text.rfind(" ", start, start + max_chars)
        if split <= start:
            split = start + max_chars  # hard cut — no whitespace in window
        else:
            split += 1  # include the space in this chunk
        chunks.append((start, text[start:split]))
        start = split
    return chunks


def _to_entity(raw: dict, original: str) -> PIIEntity:
    start, end = _tighten(int(raw["start"]), int(raw["end"]), original)
    label = raw.get("entity_group") or raw["entity"]
    return PIIEntity(
        label=label,
        text=original[start:end],
        start=start,
        end=end,
        score=float(raw["score"]),
    )


def _merge_adjacent_entities(
    entities: list[PIIEntity],
    original: str,
    gap_allowed: dict[str, frozenset[str]] | None = None,
) -> list[PIIEntity]:
    if not entities:
        return entities
    ordered = sorted(entities, key=lambda e: e.start)
    merged: list[PIIEntity] = []
    for e in ordered:
        if merged:
            prev = merged[-1]
            gap = original[prev.end : e.start]
            if prev.label == e.label and _gap_mergeable(prev.label, gap, gap_allowed):
                start, end = _tighten(prev.start, e.end, original)
                merged[-1] = PIIEntity(
                    label=prev.label,
                    text=original[start:end],
                    start=start,
                    end=end,
                    score=min(prev.score, e.score),
                )
                continue
        merged.append(e)
    return merged


def _gap_mergeable(
    label: str,
    gap: str,
    gap_allowed: dict[str, frozenset[str]] | None,
) -> bool:
    """Empty gap always merges. Otherwise every character of the gap must be
    in `gap_allowed[label]` — whitespace is not a built-in pass; it only
    counts if the label's allowed set includes it.
    """
    if gap == "":
        return True
    allowed = (gap_allowed or {}).get(label)
    if not allowed:
        return False
    return all(c in allowed for c in gap)


def load_merge_gap(path: str | Path) -> dict[str, str]:
    """Parse a merge-gap config JSON file into `{label: allowed_chars}`.

    Shape: a flat JSON object mapping label -> string of allowed characters.
    Empty string is allowed and means "only empty gap merges for this label".
    Raises ValueError on bad shape.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON — {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object mapping label -> chars")
    bad = [k for k, v in data.items() if not (isinstance(k, str) and isinstance(v, str))]
    if bad:
        raise ValueError(f"{path}: non-string label or chars for keys: {bad!r}")
    return data


def _tighten(start: int, end: int, original: str) -> tuple[int, int]:
    """Shrink [start, end) to exclude leading/trailing whitespace.

    Keeps the invariant `entity.text == original[entity.start:entity.end]`,
    which the mask/unmask layer relies on for offset-correct replacement.
    """
    while start < end and original[start].isspace():
        start += 1
    while end > start and original[end - 1].isspace():
        end -= 1
    return start, end
