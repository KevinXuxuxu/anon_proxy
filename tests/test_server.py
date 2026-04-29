"""End-to-end proxy latency telemetry tests.

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

    app = build_app(masker=masker, upstream="http://stub", debug=False)

    @asynccontextmanager
    async def lifespan_override(_):
        async with httpx.AsyncClient(transport=transport, base_url="http://stub") as c:
            app.state.client = c
            app.state.masker = masker
            app.state.upstream = "http://stub"
            app.state.debug = False
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
            "/v1/messages",
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
            "/v1/messages",
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
