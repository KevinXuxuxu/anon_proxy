"""Anthropic Messages API request/response transforms.

Masked on outbound requests: `messages[*].content` text blocks,
`tool_use.input` string leaves (assistant history), and `tool_result.content`
(string or nested text blocks).

NOT masked: `system` (tool definitions and instructions — static, not user data),
`tools` (tool schemas), and `thinking` blocks (extended-thinking signatures are
computed over original text by upstream).

Unmasked on inbound responses: `text` blocks and `tool_use.input` string leaves
(non-streaming); `text_delta.text` and `input_json_delta.partial_json`
(streaming). Input-JSON deltas use JSON-escaped substitution so originals with
quotes/backslashes don't corrupt the stream.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable

from anon_proxy.masker import Masker

TextHook = Callable[[str], None]


def mask_request(body: dict, masker: Masker) -> dict:
    """Return a copy of an Anthropic Messages request body with PII masked.

    Only touches `messages[*].content` — user/assistant text, tool_use.input,
    and tool_result.content. The system prompt is left intact because it
    contains static tool definitions and instructions, not user data.

    The proxy layer owns the telemetry `request_scope`; this function does
    not open one. Callers outside the proxy can wrap their own scope.
    """
    result = dict(body)
    messages = body.get("messages")
    if isinstance(messages, list):
        result["messages"] = [_mask_message(m, masker) for m in messages]
    return result


def unmask_response(body: dict, masker: Masker, *, telemetry_batch=None, side: str = "response") -> dict:
    """Return a copy of a non-streaming Messages response with text unmasked.

    When `telemetry_batch` is provided, runs detectors on each text block before
    unmasking and feeds the detected spans into the batch tagged with `side`.
    This catches "leak-back" — raw PII the model emits that the outbound masker missed.
    """
    result = dict(body)
    content = body.get("content")
    if isinstance(content, list):
        result["content"] = [_unmask_block(b, masker, telemetry_batch=telemetry_batch, side=side) for b in content]
    return result


def _mask_message(message, masker: Masker):
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": masker.mask(content)}
    if isinstance(content, list):
        return {**message, "content": [_mask_block(b, masker) for b in content]}
    return message


def _mask_block(block, masker: Masker):
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "text" and isinstance(block.get("text"), str):
        return {**block, "text": masker.mask(block["text"])}
    if btype == "tool_use":
        input_val = block.get("input")
        if isinstance(input_val, (dict, list)):
            return {**block, "input": _walk_strings(input_val, masker.mask)}
        return block
    if btype == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return {**block, "content": masker.mask(content)}
        if isinstance(content, list):
            return {**block, "content": [_mask_block(b, masker) for b in content]}
        return block
    return block


def _unmask_block(block, masker: Masker, *, telemetry_batch=None, side: str = "response"):
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "text" and isinstance(block.get("text"), str):
        text = block["text"]
        if telemetry_batch is not None and text:
            _observe_response(text, masker, telemetry_batch, side)
        return {**block, "text": masker.unmask(text)}
    if btype == "tool_use":
        input_val = block.get("input")
        if isinstance(input_val, (dict, list)):
            return {**block, "input": _walk_strings(input_val, masker.unmask)}
        return block
    return block


def _observe_response(text: str, masker: Masker, telemetry_batch, side: str) -> None:
    """Run detectors on response text; feed spans into the batch tagged with side."""
    ml_spans, user_spans = masker.detect_only(text)
    kept = ml_spans + user_spans
    telemetry_batch.observe_v2(
        text=text, ml_spans=ml_spans, user_spans=user_spans, kept=kept, events=[],
        side=side,
    )


def _walk_strings(value, transform):
    """Apply `transform` to every string leaf of a JSON-shaped value."""
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {k: _walk_strings(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_strings(v, transform) for v in value]
    return value


_STREAM_HANDLERS: dict[str, dict] = {
    "text":     {"delta_type": "text_delta",       "field": "text",         "escape": False},
    "tool_use": {"delta_type": "input_json_delta", "field": "partial_json", "escape": True},
}


async def transform_stream(
    upstream_bytes: AsyncIterator[bytes],
    masker: Masker,
    *,
    on_upstream_text: TextHook | None = None,
    on_client_text: TextHook | None = None,
    on_unmask_us: Callable[[int], None] | None = None,
    telemetry_batch=None,
) -> AsyncIterator[bytes]:
    """Unmask masked payloads in an Anthropic SSE stream.

    Handles two block types: `text` (text_delta.text) and `tool_use`
    (input_json_delta.partial_json, substituted with JSON-escaped originals).

    Placeholders like <PERSON_1> can split across chunks (`<PER` / `SON_1>`), so
    a per-block tail buffer holds back anything that might be an incomplete
    token (a trailing `<` with no matching `>`). The buffer is flushed as an
    injected delta event immediately before each content_block_stop.

    `on_upstream_text` fires with each raw masked fragment as it arrives from
    upstream; `on_client_text` fires with each fragment emitted to the client
    (post-unmask). Both are optional hooks for debug/logging.
    `on_unmask_us` fires with the microseconds spent on each unmask call;
    the caller accumulates the total and converts to ms at commit time.
    When `telemetry_batch` is provided, runs detectors on each completed text block
    and feeds detected spans into the batch tagged as side="response".
    """

    def _timed_unmask(value: str, *, json_ctx: bool) -> str:
        if on_unmask_us is None:
            return masker.unmask_json(value) if json_ctx else masker.unmask(value)
        t = time.perf_counter()
        out = masker.unmask_json(value) if json_ctx else masker.unmask(value)
        on_unmask_us(int((time.perf_counter() - t) * 1_000_000))
        return out

    blocks: dict[int, dict] = {}
    raw = b""
    async for chunk in upstream_bytes:
        raw += chunk
        while b"\n\n" in raw:
            event_bytes, raw = raw.split(b"\n\n", 1)
            event_type, data_str = _parse_sse(event_bytes)
            for out_event, out_data in _transform_event(
                event_type, data_str, masker, blocks, on_upstream_text, on_client_text,
                _timed_unmask, telemetry_batch=telemetry_batch,
            ):
                yield _serialize_sse(out_event, out_data)
    if raw.strip():
        # Trailing bytes with no terminating blank line — pass them through as-is
        # so we don't silently drop content if the upstream misformatted.
        yield raw


def _parse_sse(event_bytes: bytes) -> tuple[str | None, str | None]:
    event_type: str | None = None
    data_parts: list[str] = []
    for line in event_bytes.decode("utf-8", errors="replace").splitlines():
        if line.startswith(":") or not line:
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            chunk = line[len("data:") :]
            if chunk.startswith(" "):
                chunk = chunk[1:]
            data_parts.append(chunk)
    data = "\n".join(data_parts) if data_parts else None
    return event_type, data


def _serialize_sse(event_type: str | None, data: str | None) -> bytes:
    lines: list[str] = []
    if event_type:
        lines.append(f"event: {event_type}")
    if data is not None:
        lines.append(f"data: {data}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


_MAX_ASSEMBLED_BYTES = 100_000  # 100 KB per-block telemetry cap


def _transform_event(
    event_type,
    data_str,
    masker: Masker,
    blocks: dict[int, dict],
    on_upstream_text: TextHook | None,
    on_client_text: TextHook | None,
    timed_unmask: Callable[[str, bool], str] | None = None,
    *,
    telemetry_batch=None,
):
    # Use timed_unmask if provided, otherwise fall back to plain _unmask_for.
    def _do_unmask(text: str, escape: bool) -> str:
        if timed_unmask is not None:
            return timed_unmask(text, json_ctx=escape)
        return _unmask_for(masker, text, escape)

    if data_str is None:
        yield event_type, None
        return
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        yield event_type, data_str
        return

    if event_type == "content_block_start":
        idx = data.get("index", 0)
        cb = data.get("content_block") or {}
        handler = _STREAM_HANDLERS.get(cb.get("type"))
        if handler:
            # "assembled" tracks full upstream text for telemetry (text blocks only).
            blocks[idx] = {**handler, "buffer": "", "assembled": ""}
            # tool_use may carry non-empty initial input — unmask it in place.
            if cb.get("type") == "tool_use":
                input_val = cb.get("input")
                if isinstance(input_val, (dict, list)) and input_val:
                    new_cb = {**cb, "input": _walk_strings(input_val, lambda v: _do_unmask(v, False))}
                    yield event_type, json.dumps({**data, "content_block": new_cb})
                    return
        yield event_type, data_str
        return

    if event_type == "content_block_delta":
        idx = data.get("index", 0)
        delta = data.get("delta") or {}
        state = blocks.get(idx)
        if state and delta.get("type") == state["delta_type"]:
            field = state["field"]
            piece = delta.get(field) or ""
            if on_upstream_text and piece:
                on_upstream_text(piece)
            # Accumulate full upstream text for response-side detection (text blocks only).
            if state["field"] == "text" and piece:
                cur = state["assembled"]
                if len(cur) < _MAX_ASSEMBLED_BYTES:
                    state["assembled"] = cur + piece
            buf = state["buffer"] + piece
            emittable, remainder = _split_emit(buf)
            state["buffer"] = remainder
            if emittable:
                unmasked = _do_unmask(emittable, state["escape"])
                if on_client_text:
                    on_client_text(unmasked)
                new_data = {**data, "delta": {**delta, field: unmasked}}
                yield event_type, json.dumps(new_data)
            return
        yield event_type, data_str
        return

    if event_type == "content_block_stop":
        idx = data.get("index", 0)
        state = blocks.pop(idx, None)
        if state and state["buffer"]:
            unmasked = _do_unmask(state["buffer"], state["escape"])
            if on_client_text:
                on_client_text(unmasked)
            flush = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": state["delta_type"], state["field"]: unmasked},
            }
            yield "content_block_delta", json.dumps(flush)
        # Run response-side detector on the assembled text block (text blocks only).
        if (
            telemetry_batch is not None
            and state is not None
            and state["field"] == "text"
        ):
            assembled = state["assembled"]
            if assembled:
                _observe_response(assembled, masker, telemetry_batch, "response")
        yield event_type, json.dumps(data)
        return

    yield event_type, data_str


def _unmask_for(masker: Masker, text: str, escape: bool) -> str:
    return masker.unmask_json(text) if escape else masker.unmask(text)


def _split_emit(buf: str) -> tuple[str, str]:
    """Split into (emittable, remainder). Remainder is anything from the last
    unterminated '<' onward — a potentially-incomplete placeholder token."""
    last_open = buf.rfind("<")
    if last_open == -1 or ">" in buf[last_open:]:
        return buf, ""
    return buf[:last_open], buf[last_open:]
