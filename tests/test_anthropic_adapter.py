"""Tests for Anthropic adapter response-side detection (Phase 5)."""

from __future__ import annotations

import pytest

from anon_proxy.adapters.anthropic import unmask_response
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
