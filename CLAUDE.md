# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early-stage / skeleton. The repository currently contains only a stub `main.py`, a `pyproject.toml` with no dependencies declared, and the design intent described in the README. Any real functionality — the privacy filter integration, the persistent dictionary, the proxy server — has not been written yet. Treat most tasks here as greenfield work rather than modification.

## Intended architecture (from README)

The project is an LLM API proxy that transparently masks PII before requests leave the device and un-masks it on the way back. Three pieces that must stay consistent with each other:

1. **Local PII detector** — runs [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) locally so raw PII never leaves the machine. This is a hard constraint: anything that would send unmasked text to a remote model breaks the core guarantee.
2. **Persistent, reversible mapping** — a dictionary of `real value ↔ placeholder` that survives across requests so the same entity gets the same placeholder every time. Reversibility is what lets the proxy un-mask the model's response.
3. **OpenAI-compatible proxy** — wraps an upstream OpenAI-compatible LLM API, applying mask on the way out and un-mask on the way in. Clients should be able to point their existing OpenAI SDK at this proxy without code changes.

When designing new code, make sure these three responsibilities remain cleanly separable — the masking layer should not know about HTTP, and the proxy layer should not know about the detector's internals.

## Toolchain

- Python `>=3.10` (pinned to `3.10` in `.python-version`).
- `pyproject.toml` is PEP 621 style with no build-backend declared yet — the pin file + minimal pyproject shape suggests [`uv`](https://docs.astral.sh/uv/) as the intended package manager. If you add dependencies, use `uv add <pkg>` so `pyproject.toml` and the lockfile stay in sync.
- Run the stub: `uv run python main.py` (or `python main.py` inside an active venv).

No tests, lint config, or CI exist yet — add them alongside the first real module rather than retrofitting later.
