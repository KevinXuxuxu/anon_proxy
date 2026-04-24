"""HTTP proxy for the Anthropic Messages API with transparent PII masking.

Only `POST /v1/messages` is transformed — the request body is masked before
forwarding, and the response (SSE or JSON) is unmasked before returning. All
other paths pass through unchanged, so `GET /v1/models`, `POST /v1/messages/batches`,
etc. continue to work.

Client auth headers (`x-api-key`, `anthropic-version`, `anthropic-beta`, ...) are
forwarded verbatim. The proxy never inspects or stores them.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from anon_proxy.adapters import anthropic as anthropic_adapter
from anon_proxy.masker import Masker
from anon_proxy.privacy_filter import PrivacyFilter, load_merge_gap
from anon_proxy.regex_detector import RegexDetector, load_patterns
from anon_proxy.telemetry import (
    DEFAULT_PATH as TELEMETRY_DEFAULT_PATH,
    JSONLWriter,
    TelemetryObserver,
    default_detector as default_telemetry_detector,
)

DEFAULT_UPSTREAM = "https://api.anthropic.com"

_DIM = "\033[2m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


def _trunc(s: str, n: int = 100) -> str:
    s = s.replace("\n", "↵")
    return repr(s if len(s) <= n else s[:n] + "…")


def _diff_messages(before: list, after: list) -> list[str]:
    lines = []
    for mb, ma in zip(before, after):
        role = mb.get("role", "?")
        cb, ca = mb.get("content"), ma.get("content")
        if isinstance(cb, str) and cb != ca:
            lines.append(f"  {role}: {_trunc(cb)} → {_trunc(ca)}")
        elif isinstance(cb, list):
            for j, (bb, ba) in enumerate(zip(cb, ca)):
                if bb == ba:
                    continue
                btype = bb.get("type", "?")
                if btype == "text":
                    lines.append(
                        f"  {role}[{j}] text: {_trunc(bb.get('text',''))} → {_trunc(ba.get('text',''))}"
                    )
                elif btype == "tool_use":
                    bi = json.dumps(bb.get("input", {}), ensure_ascii=False)
                    ai = json.dumps(ba.get("input", {}), ensure_ascii=False)
                    lines.append(
                        f"  {role}[{j}] tool_use({bb.get('name','?')}): {_trunc(bi)} → {_trunc(ai)}"
                    )
                elif btype == "tool_result":
                    bc, ac = bb.get("content", ""), ba.get("content", "")
                    if isinstance(bc, str):
                        lines.append(f"  {role}[{j}] tool_result: {_trunc(bc)} → {_trunc(ac)}")
                    else:
                        lines.append(f"  {role}[{j}] tool_result: (content changed)")
    return lines


def _diff_content(before: list, after: list) -> list[str]:
    lines = []
    for i, (bb, ba) in enumerate(zip(before, after)):
        if bb == ba:
            continue
        btype = bb.get("type", "?")
        if btype == "text":
            lines.append(f"  text[{i}]: {_trunc(bb.get('text',''))} → {_trunc(ba.get('text',''))}")
        elif btype == "tool_use":
            bi = json.dumps(bb.get("input", {}), ensure_ascii=False)
            ai = json.dumps(ba.get("input", {}), ensure_ascii=False)
            lines.append(f"  tool_use[{i}]({bb.get('name','?')}): {_trunc(bi)} → {_trunc(ai)}")
    return lines


def _log_request(incoming: dict, masked: dict, new_store_entries: list[tuple[str, str]]) -> None:
    model = incoming.get("model", "?")
    n_msg = len(incoming.get("messages", []))
    print(f"\n{_DIM}==== POST /v1/messages | model={model} | {n_msg} msg ===={_RESET}", file=sys.stderr)
    if new_store_entries:
        print(f"{_DIM}[store +{len(new_store_entries)}]{_RESET}", file=sys.stderr)
        for token, original in new_store_entries:
            print(f"  {token}  ←  {original!r}", file=sys.stderr)
    diffs = _diff_messages(incoming.get("messages", []), masked.get("messages", []))
    if diffs:
        print(f"{_YELLOW}[masked]{_RESET}", file=sys.stderr)
        for line in diffs:
            print(line, file=sys.stderr)
    elif not new_store_entries:
        print(f"{_DIM}(no PII detected){_RESET}", file=sys.stderr)
    sys.stderr.flush()


def _log_response(upstream: dict, unmasked: dict) -> None:
    diffs = _diff_content(upstream.get("content", []), unmasked.get("content", []))
    if diffs:
        print(f"{_GREEN}[unmasked response]{_RESET}", file=sys.stderr)
        for line in diffs:
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
    upstream: str | None = None,
    debug: bool = False,
) -> Starlette:
    masker = masker or Masker()
    upstream_url = (upstream or os.environ.get("ANON_PROXY_UPSTREAM") or DEFAULT_UPSTREAM).rstrip("/")

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            app.state.client = client
            app.state.masker = masker
            app.state.upstream = upstream_url
            app.state.debug = debug
            yield

    async def dispatch(request: Request) -> Response:
        if request.url.path == "/v1/messages" and request.method == "POST":
            return await _handle_messages(request)
        return await _passthrough(request)

    routes = [
        Route(
            "/{path:path}",
            dispatch,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        ),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


async def _handle_messages(request: Request) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    masker: Masker = request.app.state.masker
    upstream: str = request.app.state.upstream
    debug: bool = request.app.state.debug

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return await _passthrough(request, body_override=raw_body)

    store_before = len(masker.store)
    masked = anthropic_adapter.mask_request(body, masker)
    if debug:
        new_entries = masker.store.items()[store_before:]
        _log_request(body, masked, new_entries)

    masked_bytes = json.dumps(masked).encode("utf-8")
    upstream_headers = _forward_request_headers(request.headers)
    url = upstream + request.url.path
    params = dict(request.query_params)

    if bool(masked.get("stream")):
        req = client.build_request(
            "POST", url, content=masked_bytes, headers=upstream_headers, params=params,
        )
        upstream_resp = await client.send(req, stream=True)

        if upstream_resp.status_code >= 400:
            err_body = await upstream_resp.aread()
            await upstream_resp.aclose()
            return Response(
                content=err_body,
                status_code=upstream_resp.status_code,
                headers=_filter_response_headers(upstream_resp.headers),
                media_type=upstream_resp.headers.get("content-type"),
            )

        async def body_iter():
            upstream_buf: list[str] = []
            client_buf: list[str] = []
            try:
                async for out in anthropic_adapter.transform_stream(
                    upstream_resp.aiter_bytes(),
                    masker,
                    on_upstream_text=upstream_buf.append if debug else None,
                    on_client_text=client_buf.append if debug else None,
                ):
                    yield out
            finally:
                if debug:
                    _log_stream_summary("".join(upstream_buf), "".join(client_buf))
                await upstream_resp.aclose()

        return StreamingResponse(
            body_iter(),
            status_code=upstream_resp.status_code,
            headers=_filter_response_headers(upstream_resp.headers),
            media_type="text/event-stream",
        )

    upstream_resp = await client.post(
        url, content=masked_bytes, headers=upstream_headers, params=params,
    )
    content_type = upstream_resp.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            resp_json = upstream_resp.json()
        except ValueError:
            resp_json = None
        if resp_json is not None and upstream_resp.status_code < 400:
            unmasked = anthropic_adapter.unmask_response(resp_json, masker)
            if debug:
                _log_response(resp_json, unmasked)
            return Response(
                content=json.dumps(unmasked),
                status_code=upstream_resp.status_code,
                headers=_filter_response_headers(upstream_resp.headers),
                media_type="application/json",
            )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=_filter_response_headers(upstream_resp.headers),
        media_type=content_type or None,
    )


async def _passthrough(request: Request, *, body_override: bytes | None = None) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    upstream: str = request.app.state.upstream
    body = body_override if body_override is not None else await request.body()
    upstream_resp = await client.request(
        request.method,
        upstream + request.url.path,
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


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="anon-proxy — PII masking proxy for the Anthropic API")
    parser.add_argument("--host", default=os.environ.get("ANON_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANON_PROXY_PORT", "8080")))
    parser.add_argument("--upstream", default=os.environ.get("ANON_PROXY_UPSTREAM", DEFAULT_UPSTREAM))
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
        "--telemetry",
        action="store_true",
        default=os.environ.get("ANON_PROXY_TELEMETRY", "").lower() in ("1", "true", "yes"),
        help="Write one JSON record per mask() call to a local JSONL log (no PII content, "
             "only labels/lengths/positions). Run `python -m anon_proxy.telemetry_report` "
             "to summarize.",
    )
    parser.add_argument(
        "--telemetry-path",
        default=os.environ.get("ANON_PROXY_TELEMETRY_PATH"),
        help=f"Telemetry log path (default: {TELEMETRY_DEFAULT_PATH}). Ignored unless --telemetry is set.",
    )
    args = parser.parse_args()

    extra_detectors = []
    if args.patterns:
        try:
            patterns = load_patterns(args.patterns)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        extra_detectors.append(RegexDetector(patterns))

    pf: PrivacyFilter | None = None
    if args.merge_gap_file or args.chunk_size != 1500:
        merge_gap = None
        if args.merge_gap_file:
            try:
                merge_gap = load_merge_gap(args.merge_gap_file)
            except (OSError, ValueError) as e:
                print(f"error: {e}", file=sys.stderr)
                sys.exit(2)
        pf = PrivacyFilter(merge_gap_allowed=merge_gap, chunk_size=args.chunk_size)

    telemetry_observer: TelemetryObserver | None = None
    telemetry_path = None
    if args.telemetry:
        telemetry_path = (
            Path(args.telemetry_path) if args.telemetry_path else TELEMETRY_DEFAULT_PATH
        )
        telemetry_observer = TelemetryObserver(
            detector=default_telemetry_detector(),
            sink=JSONLWriter(telemetry_path),
            chunk_size=args.chunk_size,
        )

    masker = (
        Masker(filter=pf, extra_detectors=extra_detectors, telemetry=telemetry_observer)
        if (pf is not None or extra_detectors or telemetry_observer is not None)
        else None
    )
    app = build_app(masker=masker, upstream=args.upstream, debug=args.debug)
    print(
        f"anon-proxy listening on http://{args.host}:{args.port}\n"
        f"  upstream: {args.upstream}\n"
        f"  debug: {args.debug}\n"
        f"  patterns: {args.patterns or '(none)'}\n"
        f"  merge-gap-file: {args.merge_gap_file or '(defaults)'}\n"
        f"  telemetry: {telemetry_path or '(off)'}\n"
        f"  point your Anthropic SDK at http://{args.host}:{args.port} via base_url",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
