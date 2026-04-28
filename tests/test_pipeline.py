from anon_proxy.pipeline import AttributedSpan, OverlapEvent, ResolveResult
from anon_proxy.privacy_filter import PIIEntity


def _entity(label: str, start: int, end: int, score: float = 1.0) -> PIIEntity:
    return PIIEntity(label=label, text="x" * (end - start), start=start, end=end, score=score)


def test_attributed_span_carries_source():
    span = AttributedSpan(entity=_entity("EMAIL", 0, 5), source="ml")
    assert span.source == "ml"
    assert span.entity.label == "EMAIL"


def test_attributed_span_source_must_be_known():
    import pytest
    with pytest.raises(ValueError):
        AttributedSpan(entity=_entity("EMAIL", 0, 5), source="bogus")


def test_overlap_event_records_winner_and_loser():
    w = AttributedSpan(entity=_entity("EMAIL", 0, 10), source="ml")
    l = AttributedSpan(entity=_entity("EMAIL", 0, 5), source="user_regex")
    ev = OverlapEvent(winner=w, loser=l, reason="overlap_longer")
    assert ev.winner.source == "ml"
    assert ev.loser.source == "user_regex"
    assert ev.reason == "overlap_longer"


def test_resolve_result_holds_kept_and_events():
    span = AttributedSpan(entity=_entity("EMAIL", 0, 5), source="ml")
    r = ResolveResult(kept=[span], events=[])
    assert r.kept == [span]
    assert r.events == []
