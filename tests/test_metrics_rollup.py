from anon_proxy.metrics_rollup import update_daily_rollup, DailyRollup


def test_update_daily_rollup_counts_labels():
    state = {}
    update_daily_rollup(state, {"ts": "2026-04-29T00:00:00Z", "spans": [
        {"label": "EMAIL", "source": "ml", "kept": True, "side": "user"},
        {"label": "EMAIL", "source": "ml", "kept": True, "side": "user"},
        {"label": "PERSON", "source": "baseline", "kept": False, "side": "user"},
    ]})
    rollup = state["2026-04-29"]
    assert rollup.label_counts == {"EMAIL": 2, "PERSON": 1}


def test_update_daily_rollup_counts_leak_back():
    state = {}
    update_daily_rollup(state, {"ts": "2026-04-29T00:00:00Z", "spans": [
        {"label": "EMAIL", "source": "ml", "kept": True, "side": "response"},
    ]})
    assert state["2026-04-29"].leak_back == 1


def test_update_daily_rollup_aggregates_across_records():
    state = {}
    update_daily_rollup(state, {"ts": "2026-04-29T00:00:00Z", "spans": [{"label": "EMAIL", "source": "ml", "kept": True}]})
    update_daily_rollup(state, {"ts": "2026-04-29T01:00:00Z", "spans": [{"label": "EMAIL", "source": "ml", "kept": True}]})
    assert state["2026-04-29"].label_counts == {"EMAIL": 2}
