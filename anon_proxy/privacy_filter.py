from dataclasses import dataclass
from typing import Iterable

from transformers import pipeline


@dataclass(frozen=True)
class PIIEntity:
    label: str
    text: str
    start: int
    end: int
    score: float


class PrivacyFilter:
    """Thin wrapper around the openai/privacy-filter token classifier.

    The HF pipeline's aggregation_strategy only merges subword pieces within
    a single word. For PII masking we also want to merge *adjacent* same-label
    spans ("Alice" + "Smith" → "Alice Smith", "alice@example" + ".com" →
    "alice@example.com") so each entity maps to one placeholder downstream.
    That second merge pass is `merge_adjacent`, on by default.
    """

    MODEL_ID = "openai/privacy-filter"

    def __init__(
        self,
        *,
        aggregation_strategy: str = "simple",
        merge_adjacent: bool = True,
        device: int | str | None = None,
    ) -> None:
        self._pipe = pipeline(
            task="token-classification",
            model=self.MODEL_ID,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )
        self._merge_adjacent = merge_adjacent

    def detect(self, text: str) -> list[PIIEntity]:
        entities = [_to_entity(r, text) for r in self._pipe(text)]
        if self._merge_adjacent:
            entities = _merge_adjacent_entities(entities, text)
        return entities

    def detect_raw(self, text: str) -> list[dict]:
        """Return the pipeline's untouched per-span dicts for debugging."""
        return list(self._pipe(text))

    def detect_batch(self, texts: Iterable[str]) -> list[list[PIIEntity]]:
        texts = list(texts)
        results = self._pipe(texts) if texts else []
        out: list[list[PIIEntity]] = []
        for t, res in zip(texts, results):
            entities = [_to_entity(r, t) for r in res]
            if self._merge_adjacent:
                entities = _merge_adjacent_entities(entities, t)
            out.append(entities)
        return out


def _to_entity(raw: dict, original: str) -> PIIEntity:
    start = int(raw["start"])
    end = int(raw["end"])
    label = raw.get("entity_group") or raw["entity"]
    return PIIEntity(
        label=label,
        text=original[start:end],
        start=start,
        end=end,
        score=float(raw["score"]),
    )


def _merge_adjacent_entities(entities: list[PIIEntity], original: str) -> list[PIIEntity]:
    if not entities:
        return entities
    ordered = sorted(entities, key=lambda e: e.start)
    merged: list[PIIEntity] = []
    for e in ordered:
        if merged:
            prev = merged[-1]
            gap = original[prev.end : e.start]
            if prev.label == e.label and (gap == "" or gap.isspace()):
                merged[-1] = PIIEntity(
                    label=prev.label,
                    text=original[prev.start : e.end].strip(),
                    start=prev.start,
                    end=e.end,
                    score=min(prev.score, e.score),
                )
                continue
        merged.append(
            PIIEntity(
                label=e.label,
                text=original[e.start : e.end].strip() or e.text,
                start=e.start,
                end=e.end,
                score=e.score,
            )
        )
    return merged
