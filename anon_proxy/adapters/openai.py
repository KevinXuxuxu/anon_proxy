"""OpenAI Chat Completions API request/response transforms.

OpenAI format uses:
- Request: messages with role/content, functions/tools
- Response: choices with message content
- Streaming: SSE with delta content

Masked on outbound: messages content, function arguments, tool call inputs.
Unmasked on inbound: message content, function arguments, tool call outputs.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

from anon_proxy.masker import Masker


def mask_request(body: dict, masker: Masker) -> dict:
    """Return a copy of an OpenAI chat completions request with PII masked.

    Masked fields:
    - messages[*].content (text)
    - messages[*].tool_calls[*].function.arguments (string)
    - tools[*].function.parameters (schema)
    """
    result = dict(body)
    messages = body.get("messages")
    if isinstance(messages, list):
        result["messages"] = [_mask_message(m, masker) for m in messages]

    # Mask tool definitions
    tools = body.get("tools")
    if isinstance(tools, list):
        result["tools"] = [_mask_tool(t, masker) for t in tools]

    return result


def unmask_response(body: dict, masker: Masker) -> dict:
    """Return a copy of a non-streaming response with text unmasked."""
    result = dict(body)
    choices = body.get("choices")
    if isinstance(choices, list):
        result["choices"] = [_unmask_choice(c, masker) for c in choices]
    return result


def _mask_message(message: dict, masker: Masker) -> dict:
    """Mask a single message."""
    if not isinstance(message, dict):
        return message

    result = dict(message)

    # Mask content
    content = message.get("content")
    if isinstance(content, str):
        result["content"] = masker.mask(content)
    elif isinstance(content, list):
        # OpenAI supports array content (text + images)
        result["content"] = [_mask_content_item(c, masker) for c in content]

    # Mask tool calls
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        result["tool_calls"] = [_mask_tool_call(tc, masker) for tc in tool_calls]

    # Mask tool call_id in role=tool messages
    if message.get("role") == "tool" and isinstance(content, str):
        result["content"] = masker.mask(content)

    return result


def _mask_content_item(item: dict, masker: Masker) -> dict:
    """Mask a content item (text or image_url)."""
    if item.get("type") == "text" and isinstance(item.get("text"), str):
        return {**item, "text": masker.mask(item["text"])}
    if item.get("type") == "image_url":
        # Could mask URLs if they contain PII, but usually just skip
        return item
    return item


def _mask_tool_call(tool_call: dict, masker: Masker) -> dict:
    """Mask a tool call (function arguments)."""
    result = dict(tool_call)
    function = tool_call.get("function", {})
    if isinstance(function, dict):
        args = function.get("arguments")
        if isinstance(args, str):
            # Arguments are JSON string - mask after parsing
            try:
                args_obj = json.loads(args)
                masked = _walk_strings(args_obj, masker.mask)
                result["function"] = {**function, "arguments": json.dumps(masked)}
            except json.JSONDecodeError:
                # If not valid JSON, just mask as string
                result["function"] = {**function, "arguments": masker.mask(args)}
        elif isinstance(args, dict):
            # Arguments as object (some variations)
            result["function"] = {**function, "arguments": _walk_strings(args, masker.mask)}
    return result


def _mask_tool(tool: dict, masker: Masker) -> dict:
    """Mask a tool definition (function parameters)."""
    result = dict(tool)
    function = tool.get("function", {})
    if isinstance(function, dict):
        # Mask function description and parameter properties
        masked_func = dict(function)
        if isinstance(function.get("description"), str):
            masked_func["description"] = masker.mask(function["description"])
        if isinstance(function.get("parameters"), dict):
            masked_func["parameters"] = _walk_strings(function["parameters"], masker.mask)
        result["function"] = masked_func
    return result


def _unmask_choice(choice: dict, masker: Masker) -> dict:
    """Unmask a response choice."""
    result = dict(choice)
    message = choice.get("message")
    if isinstance(message, dict):
        result["message"] = _unmask_message(message, masker)
    return result


def _unmask_message(message: dict, masker: Masker) -> dict:
    """Unmask a response message."""
    if not isinstance(message, dict):
        return message

    result = dict(message)

    # Unmask content
    content = message.get("content")
    if isinstance(content, str):
        result["content"] = masker.unmask(content)
    elif isinstance(content, list):
        result["content"] = [_unmask_content_item(c, masker) for c in content]

    # Unmask tool calls
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        result["tool_calls"] = [_unmask_tool_call(tc, masker) for tc in tool_calls]

    return result


def _unmask_content_item(item: dict, masker: Masker) -> dict:
    """Unmask a content item."""
    if item.get("type") == "text" and isinstance(item.get("text"), str):
        return {**item, "text": masker.unmask(item["text"])}
    return item


def _unmask_tool_call(tool_call: dict, masker: Masker) -> dict:
    """Unmask a tool call."""
    result = dict(tool_call)
    function = tool_call.get("function", {})
    if isinstance(function, dict):
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args_obj = json.loads(args)
                unmasked = _walk_strings(args_obj, masker.unmask)
                result["function"] = {**function, "arguments": json.dumps(unmasked)}
            except json.JSONDecodeError:
                result["function"] = {**function, "arguments": masker.unmask(args)}
        elif isinstance(args, dict):
            result["function"] = {**function, "arguments": _walk_strings(args, masker.unmask)}
    return result


def _walk_strings(value, transform):
    """Apply `transform` to every string leaf of a JSON-shaped value."""
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {k: _walk_strings(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_strings(v, transform) for v in value]
    return value


# Stream handling for OpenAI
_STREAM_HANDLERS: dict[str, dict] = {
    "content": {"delta_key": "content", "content_field": "content", "escape": False},
    "tool_calls": {"delta_key": "tool_calls", "content_field": "arguments", "escape": True},
}


async def transform_stream(
    upstream_bytes: AsyncIterator[bytes],
    masker: Masker,
    *,
    on_substitution: Callable[[str, str], None] | None = None,
) -> AsyncIterator[bytes]:
    """Unmask masked payloads in an OpenAI SSE stream.

    OpenAI streaming format:
    - data: {"choices": [{"delta": {"content": "..."}, ...}]}
    - data: [DONE] at end

    Handles content deltas and tool_calls.function.arguments deltas.

    For content deltas, we buffer chunks to handle placeholders that may be
    split across multiple events (e.g., "<PERSON_2>" might come as "PERSON", "_", "2", ">").
    """
    tool_call_buffers: dict[int, str] = {}
    content_buffer = [""]  # Buffer for accumulating content chunks (mutable list)
    raw = b""

    async for chunk in upstream_bytes:
        raw += chunk
        while b"\n\n" in raw:
            event_bytes, raw = raw.split(b"\n\n", 1)
            event_type, data_str = _parse_sse(event_bytes)

            if data_str == "[DONE]":
                # Flush any remaining content buffer before DONE
                if content_buffer[0]:
                    buffered = content_buffer[0]
                    unmasked = masker.unmask(buffered)
                    if on_substitution and buffered != unmasked:
                        on_substitution(buffered, unmasked)
                    # Yield a synthetic event with the buffered content
                    yield _serialize_sse(event_type, json.dumps({"choices": [{"delta": {"content": unmasked}}]}))
                    content_buffer[0] = ""
                yield _serialize_sse(event_type, data_str)
                continue

            for out_event, out_data in _transform_event(
                event_type,
                data_str,
                masker,
                tool_call_buffers,
                content_buffer,
                on_substitution,
            ):
                yield _serialize_sse(out_event, out_data)

    # Flush any remaining content
    if content_buffer[0]:
        buffered = content_buffer[0]
        unmasked = masker.unmask(buffered)
        if on_substitution and buffered != unmasked:
            on_substitution(buffered, unmasked)
        yield _serialize_sse(None, json.dumps({"choices": [{"delta": {"content": unmasked}}]}))

    if raw.strip():
        yield raw


def _parse_sse(event_bytes: bytes) -> tuple[str | None, str | None]:
    """Parse an SSE event."""
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
    """Serialize an SSE event."""
    lines: list[str] = []
    if event_type:
        lines.append(f"event: {event_type}")
    if data is not None:
        lines.append(f"data: {data}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _transform_event(
    event_type: str | None,
    data_str: str | None,
    masker: Masker,
    tool_call_buffers: dict[int, str],
    content_buffer: list[str],  # Changed to mutable list to allow modification
    on_substitution: Callable[[str, str], None] | None,
):
    """Transform a single SSE event.

    Content buffering: We accumulate content chunks and emit when:
    1. Unmasking succeeds (no placeholders remain in buffer)
    2. Content starts with '<' (new placeholder starting - emit previous content)
    3. Buffer gets too long AND has no unclosed '<' (avoid unbounded buffering)
    """
    if data_str is None or data_str == "[DONE]":
        yield event_type, data_str
        return

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        yield event_type, data_str
        return

    choices = data.get("choices", [])
    if not isinstance(choices, list):
        yield event_type, data_str
        return

    transformed = False
    for choice in choices:
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            continue

        # Handle content delta
        content = delta.get("content")
        if isinstance(content, str):
            # Add to buffer
            content_buffer[0] += content
            buffered = content_buffer[0]
            unmasked = masker.unmask(buffered)

            # Only emit if unmasking succeeded completely (no placeholders left)
            # OR buffer is very long (fallback to avoid memory issues)
            should_emit = False
            if unmasked != buffered and "<" not in unmasked:
                # Unmasking succeeded - no placeholders left
                should_emit = True
            elif len(buffered) > 500:
                # Buffer is getting too long, emit anyway (fallback)
                should_emit = True

            if should_emit:
                if on_substitution and buffered != unmasked:
                    on_substitution(buffered, unmasked)
                choice["delta"]["content"] = unmasked
                content_buffer[0] = ""
                yield event_type, json.dumps(data)
            else:
                # Don't emit yet - keep buffering
                pass
            transformed = True
            break

        # If content is None, emit any buffered content
        if content is None:
            if content_buffer[0]:
                buffered = content_buffer[0]
                unmasked = masker.unmask(buffered)
                if on_substitution and buffered != unmasked:
                    on_substitution(buffered, unmasked)
                choice["delta"]["content"] = unmasked
                content_buffer[0] = ""
                yield event_type, json.dumps(data)
                transformed = True
                break
            # No content and no buffer - pass through
            yield event_type, json.dumps(data)
            transformed = True
            break

        # Handle tool_calls delta
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_index = tc.get("index", 0)
                function = tc.get("function", {})
                if isinstance(function, dict):
                    args_delta = function.get("arguments", "")
                    if isinstance(args_delta, str) and args_delta:
                        # Accumulate arguments for this tool call
                        buf = tool_call_buffers.get(tc_index, "")
                        buf += args_delta
                        tool_call_buffers[tc_index] = buf

                        # Try to emit complete JSON chunks
                        if _is_complete_json(buf):
                            unmasked = masker.unmask_json(buf)
                            if on_substitution and buf != unmasked:
                                on_substitution(buf, unmasked)
                            tc["function"]["arguments"] = unmasked
                            tool_call_buffers[tc_index] = ""
                        else:
                            # Incomplete JSON, keep masked
                            tc["function"]["arguments"] = args_delta

            yield event_type, json.dumps(data)
            transformed = True
            break

    # If no delta processing happened, pass through
    if not transformed:
        yield event_type, data_str


def _is_complete_json(s: str) -> bool:
    """Check if a string is complete JSON (balanced braces)."""
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False
