"""Tests for OpenAI adapter response-side detection (Phase 5 + streaming)."""

from __future__ import annotations

import asyncio
import json

from anon_proxy.adapters.openai import transform_stream, unmask_response
from anon_proxy.masker import Masker
from anon_proxy.regex_detector import RegexDetector
from anon_proxy.telemetry import CaptureMode, TelemetryObserver


class _NullFilter:
    """A PrivacyFilter stub that never detects anything (no ML model needed)."""
    def detect(self, text):
        return []

    def chunk_ranges(self, text):
        return [(0, len(text))]


def test_openai_unmask_response_runs_detector_on_response_spans():
    masker = Masker(
        filter=_NullFilter(),
        extra_detectors=[
            RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})
        ],
    )
    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": "Email: alice@example.com"}}
        ]
    }
    with obs.new_batch() as batch:
        unmask_response(body, masker, telemetry_batch=batch, side="response")
    rec = sink[-1]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    assert any(s["label"] == "EMAIL" for s in response_spans)


def test_openai_unmask_response_no_telemetry_back_compat():
    """unmask_response without telemetry_batch is back-compatible."""
    masker = Masker(filter=_NullFilter())
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello world."}}
        ]
    }
    result = unmask_response(body, masker)
    assert result["choices"][0]["message"]["content"] == "Hello world."


def test_openai_unmask_response_empty_choices():
    """unmask_response handles missing choices gracefully."""
    masker = Masker(filter=_NullFilter())
    body = {"object": "chat.completion", "model": "gpt-4o"}
    result = unmask_response(body, masker)
    assert result == body


def test_openai_unmask_response_array_content():
    """Array content items (text + image_url) are detected on text items only."""
    masker = Masker(
        filter=_NullFilter(),
        extra_detectors=[RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})],
    )
    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Contact alice@example.com"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            }
        ]
    }
    with obs.new_batch() as batch:
        unmask_response(body, masker, telemetry_batch=batch, side="response")
    rec = sink[-1]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    assert any(s["label"] == "EMAIL" for s in response_spans)


# ---------------------------------------------------------------------------
# Streaming response leak-back detection tests
# ---------------------------------------------------------------------------

def _make_masker_with_email_regex():
    return Masker(
        filter=_NullFilter(),
        extra_detectors=[RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})],
    )


def _openai_sse_events(*choice_data_list: dict) -> bytes:
    """Encode a sequence of OpenAI-style streaming data dicts as SSE bytes, then [DONE]."""
    parts = []
    for data in choice_data_list:
        parts.append(f"data: {json.dumps(data)}\n\n")
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


async def _collect_stream(sse_bytes: bytes, masker: Masker, **kwargs) -> list[bytes]:
    """Drive transform_stream with a single chunk and collect all output bytes."""
    async def _source():
        yield sse_bytes

    results = []
    async for chunk in transform_stream(_source(), masker, **kwargs):
        results.append(chunk)
    return results


def test_openai_streaming_runs_detector_at_done():
    """OpenAI streaming: assembled choice content is fed to the detector at [DONE]."""
    masker = _make_masker_with_email_regex()
    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    sse = _openai_sse_events(
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Email me at "}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "alice@example.com"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )

    with obs.new_batch() as batch:
        asyncio.run(_collect_stream(sse, masker, telemetry_batch=batch))

    assert sink, "Expected a telemetry record"
    rec = sink[-1]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    assert len(response_spans) >= 1, f"Expected response spans, got: {rec['spans']}"
    assert any(s["label"] == "EMAIL" for s in response_spans)


def test_openai_streaming_no_telemetry_batch_unchanged():
    """transform_stream without telemetry_batch produces output without crashing."""
    masker = Masker(filter=_NullFilter())
    sse = _openai_sse_events(
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )

    chunks = asyncio.run(_collect_stream(sse, masker))
    assert len(chunks) > 0


def test_openai_streaming_none_content_delta_not_accumulated():
    """None content deltas (e.g. role-only events) are skipped in accumulation."""
    masker = _make_masker_with_email_regex()
    observe_calls = []
    original = masker.detect_only
    def counting_detect(text):
        observe_calls.append(text)
        return original(text)
    masker.detect_only = counting_detect

    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    # Only one content delta with actual text; role-only event has None content.
    sse = _openai_sse_events(
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )

    with obs.new_batch() as batch:
        asyncio.run(_collect_stream(sse, masker, telemetry_batch=batch))

    # detect_only should only have been called once (at [DONE] for choice 0).
    # The role-only delta does not add to the accumulator.
    texts_observed = [t for t in observe_calls if t]
    assert len(texts_observed) == 1
    assert texts_observed[0] == "hello"
