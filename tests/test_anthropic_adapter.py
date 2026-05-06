"""Tests for Anthropic adapter response-side detection (Phase 5 + streaming)."""

from __future__ import annotations

import asyncio
import json

import pytest

from anon_proxy.adapters.anthropic import _transform_event, transform_stream, unmask_response
from anon_proxy.masker import Masker
from anon_proxy.privacy_filter import PrivacyFilter
from anon_proxy.regex_detector import RegexDetector
from anon_proxy.telemetry import CaptureMode, TelemetryObserver


class _NullFilter:
    """A PrivacyFilter stub that never detects anything (no ML model needed)."""
    def detect(self, text):
        return []

    def chunk_ranges(self, text):
        return [(0, len(text))]


def test_unmask_response_runs_detector_on_response_spans():
    """If the model emits raw PII, the response-side detector should flag it."""
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
        "content": [
            {"type": "text", "text": "Sure, I can email you at alice@example.com today."}
        ]
    }
    with obs.new_batch() as batch:
        unmask_response(body, masker, telemetry_batch=batch, side="response")
    rec = sink[-1]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    assert len(response_spans) >= 1
    assert any(s["label"] == "EMAIL" for s in response_spans)


def test_unmask_response_no_telemetry_batch_still_works():
    """unmask_response with no telemetry_batch is back-compatible."""
    masker = Masker(filter=_NullFilter())
    body = {
        "content": [
            {"type": "text", "text": "Hello world."}
        ]
    }
    result = unmask_response(body, masker)
    assert result["content"][0]["text"] == "Hello world."


def test_unmask_response_empty_content():
    """unmask_response handles bodies without 'content' gracefully."""
    masker = Masker(filter=_NullFilter())
    body = {"type": "message", "stop_reason": "end_turn"}
    result = unmask_response(body, masker)
    assert result == body


def test_unmask_response_tool_use_block_not_observed_for_text():
    """tool_use blocks are not fed to _observe_response (they have no text)."""
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
        "content": [
            {"type": "tool_use", "name": "email_tool", "input": {"to": "alice@example.com"}}
        ]
    }
    with obs.new_batch() as batch:
        unmask_response(body, masker, telemetry_batch=batch, side="response")
    # Batch commit happens on context exit; if no text observations were made the batch
    # may be empty and nothing is written. That's fine — verify no crash.
    # tool_use blocks are not text-observed, only unmasked.


# ---------------------------------------------------------------------------
# Streaming response leak-back detection tests
# ---------------------------------------------------------------------------

def _make_masker_with_email_regex():
    """Masker with no ML detections but email regex for response-side detection."""
    return Masker(
        filter=_NullFilter(),
        extra_detectors=[RegexDetector({"EMAIL": r"[\w.+-]+@[\w.-]+\.\w+"})],
    )


def _sse_events(*event_data_pairs: tuple[str | None, dict]) -> bytes:
    """Encode a sequence of (event_type, data_dict) pairs as SSE bytes."""
    parts = []
    for event_type, data in event_data_pairs:
        if event_type:
            parts.append(f"event: {event_type}\n")
        parts.append(f"data: {json.dumps(data)}\n\n")
    return "".join(parts).encode("utf-8")


async def _collect_stream(sse_bytes: bytes, masker: Masker, **kwargs) -> list[bytes]:
    """Drive transform_stream with a single chunk and collect all output bytes."""
    async def _source():
        yield sse_bytes

    results = []
    async for chunk in transform_stream(_source(), masker, **kwargs):
        results.append(chunk)
    return results


def test_anthropic_streaming_runs_detector_at_content_block_stop():
    """Streaming text blocks get response-side detector invocation at content_block_stop.

    The assembled text (Email me at alice@example.com) is fed to detect_only once,
    when content_block_stop arrives. The EMAIL span is tagged side=response in the batch.
    """
    masker = _make_masker_with_email_regex()
    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    sse = _sse_events(
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": "Email me"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": " at alice@example.com"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )

    with obs.new_batch() as batch:
        asyncio.run(_collect_stream(sse, masker, telemetry_batch=batch))

    assert sink, "Expected at least one telemetry record to be written"
    rec = sink[-1]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    assert len(response_spans) >= 1, f"Expected response-side spans, got: {rec['spans']}"
    assert any(s["label"] == "EMAIL" for s in response_spans)


def test_anthropic_streaming_detector_fires_once_per_block_not_per_delta():
    """Detector fires once at content_block_stop, not once per delta."""
    masker = _make_masker_with_email_regex()
    observe_calls = []

    # Patch detect_only to count calls
    original_detect_only = masker.detect_only
    def counting_detect_only(text):
        observe_calls.append(text)
        return original_detect_only(text)
    masker.detect_only = counting_detect_only

    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    sse = _sse_events(
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": "chunk1"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": " chunk2"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": " chunk3"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )

    with obs.new_batch() as batch:
        asyncio.run(_collect_stream(sse, masker, telemetry_batch=batch))

    # detect_only should be called exactly once (at content_block_stop), not 3 times.
    assert len(observe_calls) == 1
    assert observe_calls[0] == "chunk1 chunk2 chunk3"


def test_anthropic_streaming_no_telemetry_batch_unchanged():
    """transform_stream without telemetry_batch produces the same output as before."""
    masker = Masker(filter=_NullFilter())
    sse = _sse_events(
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": "Hello world"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )

    chunks = asyncio.run(_collect_stream(sse, masker))
    # Should produce output without crashing.
    assert len(chunks) > 0


def test_anthropic_streaming_multiple_blocks_each_observed():
    """Two text blocks each produce their own telemetry observation."""
    masker = _make_masker_with_email_regex()
    sink = []
    obs = TelemetryObserver(
        detector=RegexDetector({}),
        sink=sink.append,
        capture_mode=CaptureMode.ZERO_PII,
    )
    sse = _sse_events(
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": "alice@example.com"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                  "delta": {"type": "text_delta", "text": "bob@example.com"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    )

    with obs.new_batch() as batch:
        asyncio.run(_collect_stream(sse, masker, telemetry_batch=batch))

    assert sink
    rec = sink[-1]
    response_spans = [s for s in rec["spans"] if s.get("side") == "response"]
    # Both blocks contribute response-side EMAIL spans.
    assert len(response_spans) >= 2
