"""HTTP proxy for LLM APIs with transparent PII masking and multi-provider support.

Routes requests based on provider prefix in the URL path:
  /{provider}/{api-path} -> {provider-base-url}/{api-path}

Examples:
  /anthropic/v1/messages      -> https://api.anthropic.com/v1/messages
  /openai/v1/chat/completions -> https://api.openai.com/v1/chat/completions
  /zai/v1/messages            -> https://api.z.ai/api/anthropic/v1/messages

The proxy is stateless - each request uses the provider specified in the URL path.
Client auth headers are forwarded verbatim and never stored.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Mount, Route

from anon_proxy.adapters import anthropic as anthropic_adapter
from anon_proxy.adapters import openai as openai_adapter
from anon_proxy.crypto import KeyNotFoundError, ensure_key_exists, resolve_key
from anon_proxy.masker import Masker
from anon_proxy.privacy_filter import PrivacyFilter, load_merge_gap
from anon_proxy.regex_detector import RegexDetector, load_patterns
from anon_proxy.storage_paths import exclude_from_time_machine, is_under_sync_root, secure_create_dir
from anon_proxy.telemetry import (
    DEFAULT_PATH as TELEMETRY_DEFAULT_PATH,
    CaptureMode,
    JSONLWriter,
    TelemetryObserver,
    default_detector as default_telemetry_detector,
)
from anon_proxy.upstream import BUILT_IN_UPSTREAMS, UpstreamConfig, get_upstream_config

_DIM = "\033[2m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RESET = "\033[0m"

# Adapter registry
_ADAPTERS = {
    "anthropic": anthropic_adapter,
    "openai": openai_adapter,
}


def _trunc(s: str, n: int = 100) -> str:
    s = s.replace("\n", "↵")
    return repr(s if len(s) <= n else s[:n] + "…")


def _log_request(provider: str, path: str, incoming: dict, masked: dict, new_store_entries: list[tuple[str, str]]) -> None:
    model = incoming.get("model", "?")
    n_msg = len(incoming.get("messages", []))
    print(f"\n{_DIM}==== {provider} {path} | model={model} | {n_msg} msg ===={_RESET}", file=sys.stderr)
    if new_store_entries:
        print(f"{_DIM}[store +{len(new_store_entries)}]{_RESET}", file=sys.stderr)
        for token, original in new_store_entries:
            print(f"  {token}  ←  {original!r}", file=sys.stderr)
    diffs = _diff_content(incoming, masked)
    if diffs:
        print(f"{_YELLOW}[masked]{_RESET}", file=sys.stderr)
        for line in diffs:
            print(line, file=sys.stderr)
    elif not new_store_entries:
        print(f"{_DIM}(no PII detected){_RESET}", file=sys.stderr)
    sys.stderr.flush()


def _diff_content(before: dict, after: dict) -> list[str]:
    """Compare request/response content for diffs."""
    lines = []

    # Compare messages
    before_msg = before.get("messages", [])
    after_msg = after.get("messages", [])
    for bm, am in zip(before_msg, after_msg):
        role = bm.get("role", "?")
        bc, ac = bm.get("content"), am.get("content")

        if isinstance(bc, str) and bc != ac:
            lines.append(f"  {role}: {_trunc(bc)} → {_trunc(ac)}")
        elif isinstance(bc, list):
            for j, (bb, ba) in enumerate(zip(bc, ac)):
                if bb == ba:
                    continue
                btype = bb.get("type", "?")
                if btype == "text":
                    lines.append(f"  {role}[{j}] text: {_trunc(bb.get('text',''))} → {_trunc(ba.get('text',''))}")
                elif btype == "tool_use" or btype == "tool":
                    bi = json.dumps(bb.get("input", {}), ensure_ascii=False)
                    ai = json.dumps(ba.get("input", {}), ensure_ascii=False)
                    lines.append(f"  {role}[{j}] tool: {_trunc(bi)} → {_trunc(ai)}")
                elif btype == "tool_result":
                    lines.append(f"  {role}[{j}] tool_result: (content changed)")
                elif btype == "image_url":
                    pass  # Skip image URLs

    # Compare tool_calls (OpenAI format)
    if "tool_calls" in before or "tool_calls" in after:
        before_tc = before.get("tool_calls", [])
        after_tc = after.get("tool_calls", [])
        for btc, atc in zip(before_tc, after_tc):
            fn_before = btc.get("function", {}).get("arguments", "")
            fn_after = atc.get("function", {}).get("arguments", "")
            if fn_before != fn_after:
                lines.append(f"  tool_call: {_trunc(fn_before)} → {_trunc(fn_after)}")

    return lines


def _log_response(upstream: dict, unmasked: dict) -> None:
    """Log response unmasking."""
    lines = []

    # Handle Anthropic format
    content = upstream.get("content", [])
    if isinstance(content, list):
        for i, (bb, ba) in enumerate(zip(content, unmasked.get("content", []))):
            if bb == ba:
                continue
            btype = bb.get("type", "?")
            if btype == "text":
                lines.append(f"  text[{i}]: {_trunc(bb.get('text',''))} → {_trunc(ba.get('text',''))}")
            elif btype == "tool_use":
                bi = json.dumps(bb.get("input", {}), ensure_ascii=False)
                ai = json.dumps(ba.get("input", {}), ensure_ascii=False)
                lines.append(f"  tool_use[{i}]: {_trunc(bi)} → {_trunc(ai)}")

    # Handle OpenAI format
    choices = upstream.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content != unmasked.get("choices", [{}])[0].get("message", {}).get("content", ""):
                lines.append(f"  content: {_trunc(content)} → {_trunc(unmasked['choices'][0]['message']['content'])}")

    if lines:
        print(f"{_GREEN}[unmasked response]{_RESET}", file=sys.stderr)
        for line in lines:
            print(line, file=sys.stderr)
    sys.stderr.flush()


def _log_stream_summary(upstream_text: str, client_text: str) -> None:
    if upstream_text != client_text:
        print(
            f"{_GREEN}[unmasked stream]{_RESET} {_trunc(upstream_text)} → {_trunc(client_text)}",
            file=sys.stderr,
        )
        sys.stderr.flush()


_SKIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
}
_SKIP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
}


def build_app(
    masker: Masker | None = None,
    extra_upstreams: dict[str, UpstreamConfig] | None = None,
    debug: bool = False,
) -> Starlette:
    """Build the Starlette application.

    Args:
        masker: PII masker instance (created if None)
        extra_upstreams: Additional upstream providers configured via CLI
        debug: Enable debug logging
    """
    masker = masker or Masker()
    all_upstreams = {**BUILT_IN_UPSTREAMS, **(extra_upstreams or {})}

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            app.state.client = client
            app.state.masker = masker
            app.state.debug = debug
            app.state.upstreams = all_upstreams
            yield

    async def dispatch(request: Request) -> Response:
        """Dispatch request based on provider prefix."""
        # Split path into provider and rest
        path_parts = request.url.path.strip("/").split("/", 1)
        if not path_parts or not path_parts[0]:
            # Root path - return provider list
            return Response(
                content=json.dumps({
                    "providers": list(all_upstreams.keys()),
                    "usage": f"Use /{{provider}}/{{path}} to route to a provider. "
                           f"Available providers: {', '.join(sorted(all_upstreams.keys()))}"
                }, indent=2),
                media_type="application/json",
            )

        provider = path_parts[0]
        api_path = "/" + path_parts[1] if len(path_parts) > 1 else "/"

        # Get upstream config
        try:
            upstream_config = get_upstream_config(provider, extra_upstreams)
        except ValueError as e:
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )

        # Get adapter
        adapter = _ADAPTERS.get(upstream_config.adapter)
        if adapter is None:
            return Response(
                content=json.dumps({"error": f"No adapter for provider type: {upstream_config.adapter}"}),
                status_code=500,
                media_type="application/json",
            )

        return await _handle_proxy(request, upstream_config, adapter)

    routes = [
        Route(
            "/{path:path}",
            dispatch,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        ),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


async def _handle_proxy(
    request: Request,
    upstream_config: UpstreamConfig,
    adapter,
) -> Response:
    """Handle a proxied request."""
    client: httpx.AsyncClient = request.app.state.client
    masker: Masker = request.app.state.masker
    debug: bool = request.app.state.debug

    # Extract API path from request (remove provider prefix)
    path_parts = request.url.path.strip("/").split("/", 1)
    api_path = "/" + path_parts[1] if len(path_parts) > 1 else "/"

    # Build upstream URL
    upstream_url = urljoin(upstream_config.base_url.rstrip("/") + "/", upstream_config.path_prefix.strip("/"))
    upstream_url = urljoin(upstream_url.rstrip("/") + "/", api_path.lstrip("/"))

    # For non-POST/PUT/DELETE requests with no body, just proxy through
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return await _passthrough(request, upstream_url)

    raw_body = await request.body()

    # For requests with no body or non-JSON, just proxy
    if not raw_body or request.headers.get("content-type", "").startswith("multipart/form-data"):
        return await _passthrough(request, upstream_url, body_override=raw_body)

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return await _passthrough(request, upstream_url, body_override=raw_body)

    # Check if this is a request that should be masked
    # (Has content that might contain PII - messages, prompt, content, etc.)
    should_mask = _should_mask_request(request.url.path, body)

    if not should_mask:
        return await _passthrough(request, upstream_url, body_override=raw_body)

    # Mask the request — own the telemetry scope here so latency lands in the
    # same record. Manual __enter__/__exit__ so the streaming path can hand
    # the open scope off to body_iter.
    t_start = time.perf_counter()
    batch_cm = masker.request_scope()
    batch = batch_cm.__enter__()
    streaming_handed_off = False
    try:
        store_before = len(masker.store)
        t_mask = time.perf_counter()
        masked = adapter.mask_request(body, masker)
        mask_ms = int((time.perf_counter() - t_mask) * 1000)
        if debug:
            new_entries = masker.store.items()[store_before:]
            _log_request(upstream_config.name, api_path, body, masked, new_entries)

        masked_bytes = json.dumps(masked).encode("utf-8")
        upstream_headers = _forward_request_headers(request.headers)
        params = dict(request.query_params)

        is_streaming = bool(_get_streaming_flag(body))

        if is_streaming:
            response = await _stream_response(
                client=client,
                masker=masker,
                adapter=adapter,
                method=request.method,
                upstream_url=upstream_url,
                masked_bytes=masked_bytes,
                upstream_headers=upstream_headers,
                params=params,
                debug=debug,
                batch=batch,
                batch_cm=batch_cm,
                t_start=t_start,
                mask_ms=mask_ms,
            )
            streaming_handed_off = True
            return response

        return await _handle_non_streaming(
            client=client,
            masker=masker,
            adapter=adapter,
            method=request.method,
            upstream_url=upstream_url,
            masked_bytes=masked_bytes,
            upstream_headers=upstream_headers,
            params=params,
            debug=debug,
            batch=batch,
            t_start=t_start,
            mask_ms=mask_ms,
        )
    finally:
        if not streaming_handed_off:
            batch_cm.__exit__(None, None, None)


async def _handle_non_streaming(
    *,
    client: httpx.AsyncClient,
    masker: Masker,
    adapter,
    method: str,
    upstream_url: str,
    masked_bytes: bytes,
    upstream_headers: dict,
    params: dict,
    debug: bool,
    batch,
    t_start: float,
    mask_ms: int,
) -> Response:
    """Non-streaming branch: upstream call + unmask + record latency."""
    t_upstream = time.perf_counter()
    upstream_resp = await client.request(
        method, upstream_url, content=masked_bytes, headers=upstream_headers, params=params,
    )
    upstream_ms = int((time.perf_counter() - t_upstream) * 1000)
    content_type = upstream_resp.headers.get("content-type", "")

    response, unmask_ms = _build_unmasked_response(upstream_resp, content_type, masker, adapter, debug, batch=batch)

    if batch is not None:
        batch.record_latency(
            mask_ms=mask_ms,
            upstream_ms=upstream_ms,
            unmask_ms=unmask_ms,
            total_ms=int((time.perf_counter() - t_start) * 1000),
        )
    return response


def _build_unmasked_response(
    upstream_resp: httpx.Response,
    content_type: str,
    masker: Masker,
    adapter,
    debug: bool,
    *,
    batch=None,
) -> tuple[Response, int]:
    """Return (Response, unmask_ms). unmask_ms=0 when no unmask phase ran (non-JSON or 4xx/5xx)."""
    if content_type.startswith("application/json"):
        try:
            resp_json = upstream_resp.json()
        except ValueError:
            resp_json = None
        if resp_json is not None and upstream_resp.status_code < 400:
            t_unmask = time.perf_counter()
            unmasked = adapter.unmask_response(resp_json, masker, telemetry_batch=batch)
            unmask_ms = int((time.perf_counter() - t_unmask) * 1000)
            if debug:
                _log_response(resp_json, unmasked)
            return (
                Response(
                    content=json.dumps(unmasked),
                    status_code=upstream_resp.status_code,
                    headers=_filter_response_headers(upstream_resp.headers),
                    media_type="application/json",
                ),
                unmask_ms,
            )
    return (
        Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_filter_response_headers(upstream_resp.headers),
            media_type=content_type or None,
        ),
        0,
    )


async def _stream_response(
    *,
    client: httpx.AsyncClient,
    masker: Masker,
    adapter,
    method: str,
    upstream_url: str,
    masked_bytes: bytes,
    upstream_headers: dict,
    params: dict,
    debug: bool,
    batch,
    batch_cm,
    t_start: float,
    mask_ms: int,
) -> Response:
    """Streaming branch: records latency in body_iter's finally and exits the scope there."""
    req = client.build_request(
        method, upstream_url, content=masked_bytes, headers=upstream_headers, params=params,
    )
    t_upstream = time.perf_counter()
    upstream_resp = await client.send(req, stream=True)

    if upstream_resp.status_code >= 400:
        err_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        if batch is not None:
            batch.record_latency(
                mask_ms=mask_ms,
                upstream_ms=int((time.perf_counter() - t_upstream) * 1000),
                unmask_ms=0,
                total_ms=int((time.perf_counter() - t_start) * 1000),
            )
        batch_cm.__exit__(None, None, None)
        return Response(
            content=err_body,
            status_code=upstream_resp.status_code,
            headers=_filter_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def body_iter():
        upstream_buf: list[str] = []
        client_buf: list[str] = []
        unmask_us_total = 0

        def acc_unmask_us(us: int) -> None:
            nonlocal unmask_us_total
            unmask_us_total += us

        try:
            async for out in adapter.transform_stream(
                upstream_resp.aiter_bytes(),
                masker,
                on_upstream_text=upstream_buf.append if debug else None,
                on_client_text=client_buf.append if debug else None,
                on_unmask_us=acc_unmask_us,
            ):
                yield out
        finally:
            if debug:
                _log_stream_summary("".join(upstream_buf), "".join(client_buf))
            await upstream_resp.aclose()
            if batch is not None:
                batch.record_latency(
                    mask_ms=mask_ms,
                    upstream_ms=int((time.perf_counter() - t_upstream) * 1000),
                    unmask_ms=unmask_us_total // 1000,
                    total_ms=int((time.perf_counter() - t_start) * 1000),
                )
            batch_cm.__exit__(None, None, None)

    return StreamingResponse(
        body_iter(),
        status_code=upstream_resp.status_code,
        headers=_filter_response_headers(upstream_resp.headers),
        media_type="text/event-stream",
    )


def _should_mask_request(path: str, body: dict) -> bool:
    """Determine if a request should be masked.

    Requests that should be masked contain user-generated content:
    - Anthropic: POST /v1/messages
    - OpenAI: POST /v1/chat/completions, /v1/completions
    """
    # Check for common completion endpoints
    if path in ("/v1/messages", "/chat/completions"):
        return True

    # Check for content that might contain PII
    pii_fields = ["messages", "prompt", "content", "input", "text"]
    return any(field in body for field in pii_fields)


def _get_streaming_flag(body: dict) -> bool:
    """Extract streaming flag from request body."""
    return body.get("stream", False)


async def _passthrough(request: Request, upstream_url: str, *, body_override: bytes | None = None) -> Response:
    """Pass through request without masking."""
    client: httpx.AsyncClient = request.app.state.client
    body = body_override if body_override is not None else await request.body()

    upstream_resp = await client.request(
        request.method,
        upstream_url,
        content=body,
        headers=_forward_request_headers(request.headers),
        params=dict(request.query_params),
    )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=_filter_response_headers(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type") or None,
    )


def _forward_request_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _SKIP_REQUEST_HEADERS}


def _filter_response_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _SKIP_RESPONSE_HEADERS}


def _parse_extra_upstream(spec: str) -> tuple[str, UpstreamConfig]:
    """Parse an extra upstream specification.

    Format: name=base_url[;adapter=anthropic|openai][;path_prefix=/path]

    Examples:
        myprovider=https://api.example.com
        myprovider=https://api.example.com;adapter=openai
        myprovider=https://api.example.com;adapter=anthropic;path_prefix=api/v1
    """
    parts = spec.split(";")
    if "=" not in parts[0]:
        raise ValueError(f"Invalid upstream spec: {spec}")

    name, base_url = parts[0].split("=", 1)
    base_url = base_url.rstrip("/")

    adapter = "anthropic"
    path_prefix = ""

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == "adapter":
            if value not in ("anthropic", "openai"):
                raise ValueError(f"Invalid adapter: {value}")
            adapter = value
        elif key == "path_prefix":
            path_prefix = value

    return name, UpstreamConfig(
        name=name,
        base_url=base_url,
        path_prefix=path_prefix,
        adapter=adapter,
        sse=True,
    )


def build_telemetry_observer(
    *,
    store_pii: bool,
    corpus: bool,
    include_responses: bool,
    raw_path: Path,
    detector_patterns_path: Path | None = None,
    chunker=None,
) -> TelemetryObserver:
    """Construct a TelemetryObserver for the requested capture mode.

    Flag implication: include_responses → corpus → store_pii (each later
    flag implies the earlier ones). Returns a TelemetryObserver wired with
    the right writer, detector, and (when storing PII) the keyring key.

    Raises SystemExit(2) if the user requested a PII-storing mode but no
    encryption key is available. Refuse-to-start by design — we never
    persist PII in cleartext.
    """
    if include_responses:
        mode = CaptureMode.CORPUS_WITH_RESPONSES
    elif corpus:
        mode = CaptureMode.CORPUS
    elif store_pii:
        mode = CaptureMode.LEAN
    else:
        mode = CaptureMode.ZERO_PII

    key: bytes | None = None
    if mode != CaptureMode.ZERO_PII:
        try:
            key = resolve_key()
        except KeyNotFoundError as e:
            print(f"anon-proxy: {e}", file=sys.stderr)
            print(
                "Run with --telemetry-init-key once to generate and store a key.",
                file=sys.stderr,
            )
            raise SystemExit(2) from e

        sync_root = is_under_sync_root(raw_path)
        if sync_root:
            print(
                f"anon-proxy WARNING: telemetry path is under '{sync_root}'. The encrypted "
                f"blob will be replicated to that service. Consider --telemetry-path elsewhere.",
                file=sys.stderr,
            )
        secure_create_dir(raw_path.parent)
        if not exclude_from_time_machine(raw_path.parent):
            if sys.platform == "darwin":
                print(
                    "anon-proxy: tmutil addexclusion failed; encrypted telemetry may end up in "
                    "Time Machine backups.",
                    file=sys.stderr,
                )

    sink = JSONLWriter(raw_path)
    detector = (
        default_telemetry_detector()
        if detector_patterns_path is None
        else RegexDetector(load_patterns(detector_patterns_path))
    )
    return TelemetryObserver(
        detector=detector,
        sink=sink,
        chunker=chunker,
        capture_mode=mode,
        encryption_key=key,
    )


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="anon-proxy — PII masking proxy for LLM APIs")
    parser.add_argument("--host", default=os.environ.get("ANON_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANON_PROXY_PORT", "8080")))
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("ANON_PROXY_DEBUG", "").lower() in ("1", "true", "yes"),
        help="Log each request's masked body, response, and any new store entries to stderr.",
    )
    parser.add_argument(
        "--patterns",
        default=os.environ.get("ANON_PROXY_PATTERNS"),
        help="Path to a JSON file of additional regex patterns (label -> regex).",
    )
    parser.add_argument(
        "--merge-gap-file",
        default=os.environ.get("ANON_PROXY_MERGE_GAP"),
        help="Path to a JSON file of per-label merge-gap chars (label -> chars). "
             "Overrides entries in DEFAULT_MERGE_GAP_ALLOWED.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("ANON_PROXY_CHUNK_SIZE", "1500")),
        metavar="N",
        help="Max characters per chunk fed to the model (default: 1500). "
             "Lower values reduce peak GPU memory at the cost of more forward passes.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("ANON_PROXY_BACKEND", "auto"),
        choices=["auto", "cpu", "mps", "mlx"],
        help="PII detection backend (default: auto-detect best available).",
    )
    parser.add_argument(
        "--mlx-weights-cache",
        default=os.environ.get("ANON_PROXY_MLX_WEIGHTS_CACHE"),
        help="Path to cached MLX-converted weights. Generated on first use if not found.",
    )
    parser.add_argument(
        "--extra-upstream",
        action="append",
        default=[],
        metavar="NAME=URL[;adapter=anthropic|openai][;path_prefix=/PATH]",
        help="Add an extra upstream provider. Repeatable. "
             "Example: --extra-upstream myprovider=https://api.example.com;adapter=openai",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        default=os.environ.get("ANON_PROXY_TELEMETRY", "").lower() in ("1", "true", "yes"),
        help="Opt-in: write one JSON record per masked request to a local JSONL file "
             f"(default: {TELEMETRY_DEFAULT_PATH}). Records carry labels, lengths, and "
             "boundary flags only — never PII content.",
    )
    parser.add_argument(
        "--telemetry-path",
        default=os.environ.get("ANON_PROXY_TELEMETRY_PATH"),
        help=f"Override the telemetry log path (default: {TELEMETRY_DEFAULT_PATH}).",
    )
    parser.add_argument(
        "--telemetry-store-pii",
        action="store_true",
        default=os.environ.get("ANON_PROXY_TELEMETRY_STORE_PII", "").lower() in ("1", "true", "yes"),
        help="Lean mode: encrypt and store entity text + ±200 char windows. Requires keyring or env-var key.",
    )
    parser.add_argument(
        "--telemetry-corpus",
        action="store_true",
        default=os.environ.get("ANON_PROXY_TELEMETRY_CORPUS", "").lower() in ("1", "true", "yes"),
        help="Corpus mode: also store full input text (encrypted). Implies --telemetry-store-pii.",
    )
    parser.add_argument(
        "--telemetry-corpus-include-responses",
        action="store_true",
        default=os.environ.get("ANON_PROXY_TELEMETRY_CORPUS_INCLUDE_RESPONSES", "").lower() in ("1", "true", "yes"),
        help="Also store full response text. Implies --telemetry-corpus.",
    )
    parser.add_argument(
        "--telemetry-init-key",
        action="store_true",
        help="Generate a fresh telemetry encryption key in the OS keyring and exit.",
    )
    args = parser.parse_args()

    if args.telemetry_init_key:
        ensure_key_exists()
        print("Telemetry encryption key stored in keyring (service=anon-proxy, user=telemetry).", file=sys.stderr)
        print("Back up with: anon-proxy telemetry export-key", file=sys.stderr)
        return

    # Parse extra upstreams
    extra_upstreams = {}
    for spec in args.extra_upstream:
        try:
            name, config = _parse_extra_upstream(spec)
            extra_upstreams[name] = config
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)

    extra_detectors = []
    if args.patterns:
        try:
            patterns = load_patterns(args.patterns)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        extra_detectors.append(RegexDetector(patterns))

    pf: PrivacyFilter | None = None
    if args.merge_gap_file or args.chunk_size != 1500 or args.backend != "auto":
        merge_gap = None
        if args.merge_gap_file:
            try:
                merge_gap = load_merge_gap(args.merge_gap_file)
            except (OSError, ValueError) as e:
                print(f"error: {e}", file=sys.stderr)
                sys.exit(2)
        device = None if args.backend == "auto" else args.backend
        pf = PrivacyFilter(
            merge_gap_allowed=merge_gap,
            chunk_size=args.chunk_size,
            device=device,
        )

    telemetry_observer = None
    if args.telemetry or args.telemetry_store_pii or args.telemetry_corpus or args.telemetry_corpus_include_responses:
        log_path = Path(args.telemetry_path) if args.telemetry_path else TELEMETRY_DEFAULT_PATH
        telemetry_observer = build_telemetry_observer(
            store_pii=args.telemetry_store_pii,
            corpus=args.telemetry_corpus,
            include_responses=args.telemetry_corpus_include_responses,
            raw_path=log_path,
            detector_patterns_path=Path(args.patterns) if args.patterns else None,
            chunker=(pf.chunk_ranges if pf is not None else None),
        )

    masker = (
        Masker(filter=pf, extra_detectors=extra_detectors, telemetry=telemetry_observer)
        if (pf is not None or extra_detectors or telemetry_observer is not None)
        else None
    )

    app = build_app(masker=masker, extra_upstreams=extra_upstreams, debug=args.debug)

    all_providers = sorted({**BUILT_IN_UPSTREAMS, **extra_upstreams}.keys())
    backend_display = f"{args.backend}" if args.backend != "auto" else "auto-detect"
    telemetry_display = (
        str(Path(args.telemetry_path) if args.telemetry_path else TELEMETRY_DEFAULT_PATH)
        if args.telemetry
        else "(off)"
    )
    print(
        f"anon-proxy listening on http://{args.host}:{args.port}\n"
        f"  providers: {', '.join(all_providers)}\n"
        f"  debug: {args.debug}\n"
        f"  patterns: {args.patterns or '(none)'}\n"
        f"  merge-gap-file: {args.merge_gap_file or '(defaults)'}\n"
        f"  backend: {backend_display}\n"
        f"  telemetry: {telemetry_display}\n"
        f"\nUsage examples:\n"
        f"  Anthropic: base_url=http://{args.host}:{args.port}/anthropic\n"
        f"  OpenAI:   base_url=http://{args.host}:{args.port}/openai\n"
        f"  Custom:    base_url=http://{args.host}:{args.port}/{{provider}}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
