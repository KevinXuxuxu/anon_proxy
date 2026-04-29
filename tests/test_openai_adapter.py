"""Tests for OpenAI adapter response-side detection (Phase 5)."""

from __future__ import annotations

from anon_proxy.adapters.openai import unmask_response
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
