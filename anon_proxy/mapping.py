import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Placeholder:
    label: str
    index: int
    token: str


class PIIStore:
    """In-memory bidirectional map from (label, canonical value) to placeholder tokens.

    Cross-turn consistency: the same entity (modulo casing / whitespace) always
    maps to the same token for the life of this store. The reverse map preserves
    the first-seen original form so un-masking restores the user's casing.
    """

    def __init__(self) -> None:
        self._forward: dict[tuple[str, str], Placeholder] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def get_or_create(self, label: str, value: str) -> Placeholder:
        normalized_label = _placeholder_label(label)
        key = (normalized_label, _canonical(value))
        existing = self._forward.get(key)
        if existing is not None:
            return existing
        index = self._counters.get(normalized_label, 0) + 1
        self._counters[normalized_label] = index
        token = f"<{normalized_label}_{index}>"
        ph = Placeholder(label=normalized_label, index=index, token=token)
        self._forward[key] = ph
        self._reverse[token] = value
        return ph

    def original(self, token: str) -> str | None:
        return self._reverse.get(token)

    def tokens(self) -> list[str]:
        return list(self._reverse.keys())

    def items(self) -> list[tuple[str, str]]:
        return list(self._reverse.items())

    def __len__(self) -> int:
        return len(self._reverse)


_WHITESPACE = re.compile(r"\s+")


def _canonical(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _placeholder_label(label: str) -> str:
    trimmed = label[len("private_") :] if label.startswith("private_") else label
    return trimmed.upper()
