<img width="108" height="112" alt="anon" src="https://github.com/user-attachments/assets/6609f7ff-3e0b-458d-ac20-2f1b0b95ae62" />

# anon-proxy

An LLM API proxy that masks PII before requests leave your device and unmasks it in responses. The [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model runs **locally** — raw PII never reaches the upstream API.

```
your client  →  anon-proxy (mask)  →  api.anthropic.com
            ←  (unmask)
```

---

## Quick demo

```bash
# interactive chat through the mask/unmask layer (needs ANTHROPIC_API_KEY)
uv run python test_mask.py

# test the PII detector interactively
uv run python test_filter.py
uv run python test_filter.py "My name is Alice Smith, email alice@example.com"
```

---

## Prerequisites

- Python ≥ 3.10 (use [uv](https://docs.astral.sh/uv/))
- CUDA GPU recommended (≥4 GB VRAM); CPU works but is slower
- `ANTHROPIC_API_KEY` for `test_mask.py`; the proxy itself forwards client auth — no key needed on the server

```bash
uv sync        # install all dependencies
```

**Dependencies:** `torch`, `transformers` (local PII model), `starlette` + `uvicorn` (proxy server), `httpx` (upstream client), `anthropic` + `prompt-toolkit` (demo scripts).

---

## Running the proxy server

```bash
uv run python -m anon_proxy.server [options]
```

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on LAN) |
| `--port` | `8080` | Listen port |
| `--upstream` | `https://api.anthropic.com` | Upstream API |
| `--debug` | off | Log new store entries and masked/unmasked diffs to stderr |
| `--patterns <file>` | — | JSON file of extra regex detectors: `{"LABEL": "regex", ...}` |
| `--merge-gap-file <file>` | — | JSON file overriding per-label adjacency merge chars (see `merge_gap.json.example`) |
| `--chunk-size <N>` | `1500` | Max chars per model inference pass — lower values reduce peak VRAM |

All flags have `ANON_PROXY_*` env-var equivalents (`ANON_PROXY_HOST`, `ANON_PROXY_PORT`, `ANON_PROXY_UPSTREAM`, `ANON_PROXY_DEBUG=1`, `ANON_PROXY_PATTERNS`, `ANON_PROXY_MERGE_GAP`, `ANON_PROXY_CHUNK_SIZE`).

**With all config files:**
```bash
uv run python -m anon_proxy.server \
  --patterns patterns.json \
  --merge-gap-file merge_gap.json \
  --debug
```

---

## Using with Claude Code

1. Authenticate normally: `claude login` (browser OAuth) or set `ANTHROPIC_API_KEY`.
2. Point Claude Code at the proxy:
   ```bash
   ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
   ```
   Or set it permanently in `~/.zshrc` / `~/.bashrc`:
   ```bash
   export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
   ```
3. No other changes — the proxy forwards your auth headers unchanged.

**What gets protected:** every user and assistant message turn — text content, tool call inputs (`tool_use.input`), and tool results (`tool_result.content`). File contents, shell output, names, emails, paths containing PII are all masked before leaving your machine.

**What is NOT masked:** the system prompt (tool schemas and static instructions), tool definitions, and extended-thinking blocks (signatures would break).

**How it works:** PII spans get stable placeholder tokens (`<PERSON_1>`, `<EMAIL_1>`, `<ADDRESS_1>`, …) stored in a per-session dictionary. The same value always maps to the same token across turns so the model stays coherent. Responses are unmasked before reaching your client.

---

## Configuration files

| File | Purpose |
|---|---|
| `patterns.json` | Extra regex patterns for PII the ML model misses (SSNs, IPs, internal IDs) |
| `merge_gap.json` | Per-label chars allowed inside a gap when merging adjacent spans (e.g. hyphen for `PERSON` so "Jean-Luc" → one token) |

Copy from the `.example` files to get started.

---

## Next steps / roadmap

- **OpenAI-compatible adapter** — swap the Anthropic-specific SSE parser for an OpenAI adapter so any OpenAI SDK client works out of the box (ChatGPT, LangChain, etc.)
- **Third-party chat clients** — route traffic from clients like Open WebUI or anything that targets an OpenAI endpoint through the proxy (needs the OpenAI adapter above)
- **OpenRouter / multi-provider** — set `--upstream https://openrouter.ai/api` and add the OpenRouter auth header to cover non-Anthropic models
- **Persistent store** — optionally write the token↔original dictionary to disk so placeholder mappings survive server restarts and span multiple sessions
- **Streaming tool-result unmask** — the current streaming path unmasks `text_delta` and `input_json_delta`; tool results only appear in non-streaming responses today
