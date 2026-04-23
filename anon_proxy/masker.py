import re

from anon_proxy.mapping import PIIStore
from anon_proxy.privacy_filter import PrivacyFilter


class Masker:
    """Composes PrivacyFilter + PIIStore to mask outgoing text and unmask LLM replies.

    One Masker instance per conversation: the store accumulates entities across
    turns so the same PII always gets the same placeholder.
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()

    @property
    def store(self) -> PIIStore:
        return self._store

    def mask(self, text: str) -> str:
        entities = self._filter.detect(text)
        # Replace right-to-left so earlier spans' offsets stay valid.
        for e in sorted(entities, key=lambda x: x.start, reverse=True):
            token = self._store.get_or_create(e.label, e.text).token
            text = text[: e.start] + token + text[e.end :]
        return text

    def unmask(self, text: str) -> str:
        tokens = self._store.tokens()
        if not tokens:
            return text
        # Longest-first so "<PERSON_1>" can't shadow "<PERSON_10>".
        pattern = re.compile(
            "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
        )
        return pattern.sub(lambda m: self._store.original(m.group(0)) or m.group(0), text)
