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
from collections.abc import AsyncIterator, Callable

from anon_proxy.masker import Masker


def mask_request(body: dict, masker: Masker) -> dict:
    """Return a copy of an Anthropic Messages request body with PII masked.

    Only touches `messages[*].content` — user/assistant text, tool_use.input,
    and tool_result.content. The system prompt is left intact because it
    contains static tool definitions and instructions, not user data.
    """
    result = dict(body)
    messages = body.get("messages")
    if isinstance(messages, list):
        result["messages"] = [_mask_message(m, masker) for m in messages]
    return result


def unmask_response(body: dict, masker: Masker) -> dict:
    """Return a copy of a non-streaming Messages response with text unmasked."""
    result = dict(body)
    content = body.get("content")
    if isinstance(content, list):
        result["content"] = [_unmask_block(b, masker) for b in content]
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


def _unmask_block(block, masker: Masker):
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "text" and isinstance(block.get("text"), str):
        return {**block, "text": masker.unmask(block["text"])}
    if btype == "tool_use":
        input_val = block.get("input")
        if isinstance(input_val, (dict, list)):
            return {**block, "input": _walk_strings(input_val, masker.unmask)}
        return block
    return block


def _walk_strings(value, transform):
    """Apply `transform` to every string leaf of a JSON-shaped value."""
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {k: _walk_strings(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_strings(v, transform) for v in value]
    return value


# Configuration for streaming unmasking per block type.
# - delta_type: the event type that carries incremental content
# - field: the field name within the delta that contains the content
# - escape: whether to JSON-escape unmasked values (for tool_use partial JSON)
_STREAM_HANDLERS: dict[str, dict] = {
    "text":     {"delta_type": "text_delta",       "field": "text",         "escape": False},
    "tool_use": {"delta_type": "input_json_delta", "field": "partial_json", "escape": True},
}


async def transform_stream(
    upstream_bytes: AsyncIterator[bytes],
    masker: Masker,
    *,
    on_substitution: Callable[[str, str], None] | None = None,
) -> AsyncIterator[bytes]:
    """Unmask masked payloads in an Anthropic SSE stream.

    Stream flow:
    1. Accumulate raw bytes until we have complete SSE events (delimited by \n\n)
    2. Parse each event into (type, data)
    3. Transform events that contain masked content
    4. Serialize and emit transformed events

    Block handling:
    - text blocks: unmask <PERSON_1> placeholders back to original names
    - tool_use blocks: unmask with JSON escaping (since partial_json is a string)

    Placeholder splitting:
    - Placeholders can span chunk boundaries (<PERSON_1> → "<PER" + "SON_1>")
    - We buffer incomplete tokens (trailing "<" without ">") until block_stop
    - Buffer is flushed as an extra delta event before content_block_stop

    `on_substitution` fires with each (placeholder, unmasked) pair for debug logging.
    """
    # Per-block state: index -> {delta_type, field, escape, buffer}
    # buffer holds incomplete placeholder tokens that may complete in next chunk
    blocks: dict[int, dict] = {}
    raw = b""  # Accumulator for incomplete SSE events
    async for chunk in upstream_bytes:
        raw += chunk
        while b"\n\n" in raw:
            event_bytes, raw = raw.split(b"\n\n", 1)
            event_type, data_str = _parse_sse(event_bytes)
            for out_event, out_data in _transform_event(
                event_type, data_str, masker, blocks, on_substitution,
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


def _transform_event(
    event_type,
    data_str,
    masker: Masker,
    blocks: dict[int, dict],
    on_substitution: Callable[[str, str], None] | None,
):
    """Transform a single SSE event, unmasking content where needed.

    Event types handled:
    - content_block_start: Initialize block state, unmask initial tool_use input
    - content_block_delta: Unmask incremental content, buffer incomplete tokens
    - content_block_stop: Flush any buffered incomplete tokens
    - All other events: pass through unchanged

    The `blocks` dict tracks per-block state across events, keyed by block index.
    """
    if data_str is None:
        yield event_type, None
        return
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        yield event_type, data_str
        return

    if event_type == "content_block_start":
        # A new content block is starting. Set up state for tracking deltas.
        idx = data.get("index", 0)
        cb = data.get("content_block") or {}
        handler = _STREAM_HANDLERS.get(cb.get("type"))
        if handler:
            blocks[idx] = {**handler, "buffer": ""}
            # tool_use may carry non-empty initial input — unmask it in place.
            if cb.get("type") == "tool_use":
                input_val = cb.get("input")
                if isinstance(input_val, (dict, list)) and input_val:
                    new_cb = {**cb, "input": _walk_strings(input_val, masker.unmask)}
                    yield event_type, json.dumps({**data, "content_block": new_cb})
                    return
        yield event_type, data_str
        return

    if event_type == "content_block_delta":
        # Incremental content for a block. Unmask and buffer incomplete tokens.
        idx = data.get("index", 0)
        delta = data.get("delta") or {}
        state = blocks.get(idx)
        if state and delta.get("type") == state["delta_type"]:
            field = state["field"]
            piece = delta.get(field) or ""
            buf = state["buffer"] + piece
            emittable, remainder = _split_emit(buf)
            state["buffer"] = remainder
            if emittable:
                unmasked = _unmask_for(masker, emittable, state["escape"])
                if on_substitution and emittable != unmasked:
                    on_substitution(emittable, unmasked)
                new_data = {**data, "delta": {**delta, field: unmasked}}
                yield event_type, json.dumps(new_data)
            return
        yield event_type, data_str
        return

    if event_type == "content_block_stop":
        # Block is ending. Flush any buffered incomplete tokens.
        idx = data.get("index", 0)
        state = blocks.pop(idx, None)
        if state and state["buffer"]:
            unmasked = _unmask_for(masker, state["buffer"], state["escape"])
            if on_substitution and state["buffer"] != unmasked:
                on_substitution(state["buffer"], unmasked)
            # Inject an extra delta event to flush the buffer
            flush = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": state["delta_type"], state["field"]: unmasked},
            }
            yield "content_block_delta", json.dumps(flush)
        yield event_type, json.dumps(data)
        return

    # Pass through all other events unchanged
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
