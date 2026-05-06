"""End-to-end proxy latency telemetry tests and build_telemetry_observer factory tests.

Uses Starlette's TestClient against a stub upstream wired via httpx.ASGITransport
so neither real network nor a real ML model is needed. Verifies that one
telemetry record is emitted per request and that the record carries
`latency_ms` with all four phases.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from anon_proxy.mapping import PIIStore
from anon_proxy.masker import Masker
from anon_proxy.server import build_app
from anon_proxy.telemetry import JSONLWriter, TelemetryObserver, default_detector
from anon_proxy.upstream import UpstreamConfig


class _DummyFilter:
    """Replaces PrivacyFilter for tests — finds nothing."""

    def detect(self, text: str) -> list:
        return []

    def chunk_ranges(self, text: str) -> list:
        return [(0, len(text))]


def _make_proxy(tmp_path: Path, stub_routes: list[Route]) -> tuple[Starlette, Path]:
    log = tmp_path / "tel.jsonl"
    obs = TelemetryObserver(default_detector(), JSONLWriter(log))
    masker = Masker(filter=_DummyFilter(), store=PIIStore(), telemetry=obs)

    upstream = Starlette(routes=stub_routes)
    transport = httpx.ASGITransport(app=upstream)

    # Override the built-in `anthropic` provider to point at the in-memory stub.
    extra_upstreams = {
        "anthropic": UpstreamConfig(
            name="anthropic", base_url="http://stub", adapter="anthropic", sse=True,
        ),
    }
    app = build_app(masker=masker, extra_upstreams=extra_upstreams, debug=False)

    @asynccontextmanager
    async def lifespan_override(_):
        async with httpx.AsyncClient(transport=transport, base_url="http://stub") as c:
            app.state.client = c
            app.state.masker = masker
            app.state.debug = False
            app.state.upstreams = extra_upstreams
            yield

    app.router.lifespan_context = lifespan_override
    return app, log


def test_non_streaming_records_latency(tmp_path: Path):
    async def stub(request):
        return JSONResponse(
            content={
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
            },
            status_code=200,
        )

    app, log = _make_proxy(tmp_path, [Route("/v1/messages", stub, methods=["POST"])])

    with TestClient(app) as client:
        resp = client.post(
            "/anthropic/v1/messages",
            json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 200
    assert log.exists()
    rec = json.loads(log.read_text().strip())
    assert "latency_ms" in rec, rec
    for phase in ("mask", "upstream", "unmask", "total"):
        assert isinstance(rec["latency_ms"][phase], int)
    assert rec["latency_ms"]["total"] >= rec["latency_ms"]["mask"]
    assert rec["latency_ms"]["total"] >= rec["latency_ms"]["upstream"]


def test_streaming_records_latency(tmp_path: Path):
    """Streaming path records latency in body_iter's finally."""

    sse_body = (
        b"event: message_start\n"
        b'data: {"type":"message_start"}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )

    async def stub(request):
        async def gen():
            yield sse_body

        return StreamingResponse(gen(), media_type="text/event-stream")

    app, log = _make_proxy(tmp_path, [Route("/v1/messages", stub, methods=["POST"])])

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/anthropic/v1/messages",
            json={
                "model": "claude-x",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            for _ in resp.iter_bytes():
                pass

    assert log.exists()
    rec = json.loads(log.read_text().strip())
    assert "latency_ms" in rec, rec
    for phase in ("mask", "upstream", "unmask", "total"):
        assert phase in rec["latency_ms"]
        assert isinstance(rec["latency_ms"][phase], int)


# ---------------------------------------------------------------------------
# build_telemetry_observer factory tests (Phase 4, Task 4.1 + 4.2)
# ---------------------------------------------------------------------------

from pathlib import Path
import pytest
from anon_proxy.server import build_telemetry_observer
from anon_proxy.telemetry import CaptureMode


def test_build_telemetry_observer_zero_pii_default(tmp_path):
    obs = build_telemetry_observer(
        store_pii=False, corpus=False, include_responses=False,
        raw_path=tmp_path / "raw.jsonl",
    )
    assert obs._capture_mode == CaptureMode.ZERO_PII
    assert obs._encryption_key is None


def test_build_telemetry_observer_lean_uses_keyring(tmp_path, fake_keyring, monkeypatch):
    from anon_proxy.crypto import generate_key, store_key
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    store_key(generate_key())
    obs = build_telemetry_observer(
        store_pii=True, corpus=False, include_responses=False,
        raw_path=tmp_path / "raw.jsonl",
    )
    assert obs._capture_mode == CaptureMode.LEAN
    assert obs._encryption_key is not None


def test_build_telemetry_observer_lean_refuses_without_key(tmp_path, fake_keyring, monkeypatch):
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    # fake_keyring is empty → resolve_key raises → SystemExit
    with pytest.raises(SystemExit):
        build_telemetry_observer(
            store_pii=True, corpus=False, include_responses=False,
            raw_path=tmp_path / "raw.jsonl",
        )


def test_build_telemetry_observer_corpus_implies_store_pii(tmp_path, fake_keyring, monkeypatch):
    from anon_proxy.crypto import generate_key, store_key
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    store_key(generate_key())
    obs = build_telemetry_observer(
        store_pii=False, corpus=True, include_responses=False,
        raw_path=tmp_path / "raw.jsonl",
    )
    assert obs._capture_mode == CaptureMode.CORPUS


def test_build_telemetry_observer_include_responses_implies_corpus(tmp_path, fake_keyring, monkeypatch):
    from anon_proxy.crypto import generate_key, store_key
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    store_key(generate_key())
    obs = build_telemetry_observer(
        store_pii=False, corpus=False, include_responses=True,
        raw_path=tmp_path / "raw.jsonl",
    )
    assert obs._capture_mode == CaptureMode.CORPUS_WITH_RESPONSES


def test_telemetry_path_warns_when_under_sync_root(monkeypatch, tmp_path, capsys, fake_keyring):
    from anon_proxy.crypto import generate_key, store_key
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    store_key(generate_key())
    monkeypatch.setenv("HOME", str(tmp_path))
    raw_path = tmp_path / "Dropbox" / "anon-proxy" / "raw.jsonl"
    build_telemetry_observer(
        store_pii=True, corpus=False, include_responses=False, raw_path=raw_path,
    )
    err = capsys.readouterr().err
    assert "Dropbox" in err
    assert "encrypted" in err.lower()


def test_build_telemetry_observer_uses_raw_writer(tmp_path, fake_keyring, monkeypatch):
    from anon_proxy.crypto import generate_key, store_key
    from anon_proxy.retention import RawWriter
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    store_key(generate_key())
    obs = build_telemetry_observer(
        store_pii=True, corpus=False, include_responses=False,
        raw_path=tmp_path / "telemetry-raw.jsonl",
    )
    # The sink is a bound method of RawWriter.write
    assert isinstance(obs._sink.__self__, RawWriter)


def test_build_telemetry_observer_uses_raw_writer_for_zero_pii(tmp_path, monkeypatch):
    """ZERO_PII mode should also use RawWriter so TTL/size flags work uniformly."""
    from anon_proxy.retention import RawWriter
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    obs = build_telemetry_observer(
        store_pii=False, corpus=False, include_responses=False,
        raw_path=tmp_path / "telemetry-raw.jsonl",
    )
    # Sink is bound RawWriter.write
    assert isinstance(obs._sink.__self__, RawWriter)


def test_build_telemetry_observer_zero_pii_no_keychain_required(tmp_path, monkeypatch, fake_keyring):
    """Sanity: even with the unified writer, ZERO_PII still does not require a key."""
    monkeypatch.delenv("ANON_PROXY_TELEMETRY_KEY", raising=False)
    # fake_keyring is empty — must not raise
    obs = build_telemetry_observer(
        store_pii=False, corpus=False, include_responses=False,
        raw_path=tmp_path / "telemetry-raw.jsonl",
    )
    assert obs._encryption_key is None
