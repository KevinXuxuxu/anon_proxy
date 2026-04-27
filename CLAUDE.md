# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An LLM API proxy that transparently masks PII before requests leave the device and unmasks it in responses. The OpenAI Privacy Filter model runs locally — raw PII never reaches the upstream API.

## Commands

```bash
# Install dependencies
uv sync

# Test the PII detector interactively
uv run python test_filter.py "Alice Smith called from 555-867-5309"

# Interactive chat through the mask/unmask layer (needs ANTHROPIC_API_KEY)
uv run python test_mask.py

# Run the proxy server
uv run python -m anon_proxy.server [options]
# or
uv run python main.py [options]
```

## Architecture

The codebase is organized into four core responsibilities that remain cleanly separable:

1. **`privacy_filter.py`** — Local PII detection using the OpenAI Privacy Filter model (HuggingFace). Handles chunking for long texts, adjacency merging for multi-word entities, and configurable per-label merge gap rules.

2. **`regex_detector.py`** — Supplementary regex-based PII detector for patterns the ML model misses (SSNs, IPs, etc.). Loaded from optional `--patterns` JSON file.

3. **`mapping.py` + `masker.py`** — Persistent bidirectional mapping (`PIIStore`) and masking orchestration. Same entity gets same placeholder across requests. The `Masker` composes the PrivacyFilter with any extra detectors and handles overlap resolution.

4. **`server.py` + `adapters/`** — HTTP proxy (Starlette/Uvicorn) that applies mask on outbound and unmask on inbound. Currently Anthropic-specific; OpenAI adapter is planned (see README roadmap).

Key design invariants:
- Masking layer should not know about HTTP
- Proxy layer should not know about detector internals
- Adapters isolate provider-specific protocol details (SSE parsing, message shape)

## Configuration

Server flags (all have `ANON_PROXY_*` env var equivalents):
- `--host` / `--port` — bind address
- `--upstream` — target API URL (default: Anthropic)
- `--debug` — log masked/unmasked diffs to stderr
- `--patterns <file>` — JSON file of extra regex detectors
- `--merge-gap-file <file>` — per-label adjacency merge chars
- `--chunk-size <N>` — max chars per model inference pass (default: 1500)

## Toolchain

- Python `>=3.10` (pinned in `.python-version`)
- `uv` as package manager — use `uv add <pkg>` for dependencies
- `uvicorn` for server (ASGI)
- `transformers` + `torch` for local PII model
- No tests, lint, or CI yet — add alongside first real module
