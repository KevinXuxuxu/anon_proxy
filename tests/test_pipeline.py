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


from anon_proxy.pipeline import GreedyLongerWins


def _aspan(label: str, start: int, end: int, source: str = "ml", score: float = 1.0):
    return AttributedSpan(entity=_entity(label, start, end, score), source=source)


def test_greedy_keeps_disjoint_spans():
    a = _aspan("EMAIL", 0, 5)
    b = _aspan("PHONE", 10, 20)
    r = GreedyLongerWins().resolve([b, a])
    assert [s.start for s in r.kept] == [0, 10]
    assert r.events == []


def test_greedy_treats_touching_as_disjoint():
    a = _aspan("EMAIL", 0, 5)
    b = _aspan("EMAIL", 5, 10)
    r = GreedyLongerWins().resolve([a, b])
    assert len(r.kept) == 2
    assert r.events == []


def test_greedy_longer_wins_records_event():
    short = _aspan("EMAIL", 0, 5, source="user_regex")
    long_ = _aspan("EMAIL", 0, 10, source="ml")
    r = GreedyLongerWins().resolve([short, long_])
    assert [s.start for s in r.kept] == [0]
    assert r.kept[0].length == 10
    assert len(r.events) == 1
    assert r.events[0].winner.source == "ml"
    assert r.events[0].loser.source == "user_regex"
    assert r.events[0].reason == "overlap_longer"


def test_greedy_score_breaks_length_tie():
    ml = _aspan("EMAIL", 0, 5, source="ml", score=0.7)
    rx = _aspan("EMAIL", 0, 5, source="user_regex", score=1.0)
    r = GreedyLongerWins().resolve([ml, rx])
    assert r.kept[0].source == "user_regex"
    assert r.events[0].winner.source == "user_regex"
    assert r.events[0].loser.source == "ml"
    assert r.events[0].reason == "overlap_score_tie"


def test_greedy_empty_input():
    r = GreedyLongerWins().resolve([])
    assert r.kept == []
    assert r.events == []
