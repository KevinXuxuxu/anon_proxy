"""Anthropic Messages API request/response transforms.

Only text fields are touched. Tool definitions, tool_use inputs, and tool_result
content pass through untouched for v1 — a caller that hides PII inside a tool
argument will leak it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

from anon_proxy.masker import Masker

TextHook = Callable[[str], None]


def mask_request(body: dict, masker: Masker) -> dict:
    """Return a copy of an Anthropic Messages request body with PII masked.

    Touches `system` (string or list of text blocks) and `messages[*].content`
    (string or list of blocks; only blocks with `type == "text"` are masked).
    """
    result = dict(body)
    system = body.get("system")
    if system is not None:
        result["system"] = _mask_system(system, masker)
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


def _mask_system(system, masker: Masker):
    if isinstance(system, str):
        return masker.mask(system)
    if isinstance(system, list):
        return [_mask_block(b, masker) for b in system]
    return system


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
    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
        return {**block, "text": masker.mask(block["text"])}
    return block


def _unmask_block(block, masker: Masker):
    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
        return {**block, "text": masker.unmask(block["text"])}
    return block


async def transform_stream(
    upstream_bytes: AsyncIterator[bytes],
    masker: Masker,
    *,
    on_upstream_text: TextHook | None = None,
    on_client_text: TextHook | None = None,
) -> AsyncIterator[bytes]:
    """Unmask text_delta payloads in an Anthropic SSE stream.

    Placeholders like <PERSON_1> can split across chunks (`<PER` / `SON_1>`), so
    a per-block tail buffer holds back anything that might be an incomplete
    token (a trailing `<` with no matching `>`). The buffer is flushed as an
    injected delta event immediately before each content_block_stop.

    `on_upstream_text` fires with each raw text-delta fragment as it arrives
    from upstream; `on_client_text` fires with each fragment emitted to the
    client (post-unmask). Both are optional hooks for debug/logging.
    """
    buffers: dict[int, str] = {}
    raw = b""
    async for chunk in upstream_bytes:
        raw += chunk
        while b"\n\n" in raw:
            event_bytes, raw = raw.split(b"\n\n", 1)
            event_type, data_str = _parse_sse(event_bytes)
            for out_event, out_data in _transform_event(
                event_type, data_str, masker, buffers, on_upstream_text, on_client_text,
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
    buffers: dict[int, str],
    on_upstream_text: TextHook | None,
    on_client_text: TextHook | None,
):
    if data_str is None:
        yield event_type, None
        return
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        yield event_type, data_str
        return

    if event_type == "content_block_delta":
        delta = data.get("delta") or {}
        idx = data.get("index", 0)
        if delta.get("type") == "text_delta":
            piece = delta.get("text") or ""
            if on_upstream_text and piece:
                on_upstream_text(piece)
            buf = buffers.get(idx, "") + piece
            emittable, remainder = _split_emit(buf)
            buffers[idx] = remainder
            if emittable:
                unmasked = masker.unmask(emittable)
                if on_client_text:
                    on_client_text(unmasked)
                new_data = {**data, "delta": {**delta, "text": unmasked}}
                yield event_type, json.dumps(new_data)
            return

    if event_type == "content_block_stop":
        idx = data.get("index", 0)
        remainder = buffers.get(idx, "")
        if remainder:
            buffers[idx] = ""
            unmasked = masker.unmask(remainder)
            if on_client_text:
                on_client_text(unmasked)
            flush = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": unmasked},
            }
            yield "content_block_delta", json.dumps(flush)
        yield event_type, json.dumps(data)
        return

    yield event_type, data_str


def _split_emit(buf: str) -> tuple[str, str]:
    """Split into (emittable, remainder). Remainder is anything from the last
    unterminated '<' onward — a potentially-incomplete placeholder token."""
    last_open = buf.rfind("<")
    if last_open == -1 or ">" in buf[last_open:]:
        return buf, ""
    return buf[:last_open], buf[last_open:]
