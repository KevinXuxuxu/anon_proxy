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
# test the PII detector interactively
uv run python test_filter.py "Alice Smith called from 555-867-5309, email alice@company.com"
```
```
[PERSON:Alice Smith] called from [PHONE:555-867-5309], email [EMAIL:alice@company.com]

  PERSON       'Alice Smith'          score=0.999  offset=0-11
  PHONE        '555-867-5309'         score=0.997  offset=24-36
  EMAIL        'alice@company.com'    score=0.999  offset=45-62
```

```bash
# interactive chat through the mask/unmask layer (needs ANTHROPIC_API_KEY)
uv run python test_mask.py
```
```
you[1]> My name is Alice Smith. Summarize this note from bob@acme.com.
  sending -> My name is <PERSON_1>. Summarize this note from <EMAIL_1>.

claude[1]> Sure <PERSON_1>, here's the summary of the note from <EMAIL_1>: ...
  rendered -> Sure Alice Smith, here's the summary of the note from bob@acme.com: ...
```

---

## Prerequisites

- Python ≥ 3.10 (use [uv](https://docs.astral.sh/uv/))
- CUDA GPU recommended (≥4 GB VRAM); CPU works but is slower
- Apple Silicon (M1/M2/M3/M4) supported via MPS or MLX backends
- `ANTHROPIC_API_KEY` for `test_mask.py`; the proxy itself forwards client auth — no key needed on the server

```bash
uv sync        # install dependencies
uv sync --extra mlx  # optional: Apple Silicon MLX support
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
| `--backend` | `auto` | PII detection backend (`auto`, `cpu`, `mps`, `mlx`) |
| `--debug` | off | Log new store entries and masked/unmasked diffs to stderr |
| `--patterns <file>` | — | JSON file of extra regex detectors: `{"LABEL": "regex", ...}` |
| `--merge-gap-file <file>` | — | JSON file overriding per-label adjacency merge chars (see `merge_gap.json.example`) |
| `--chunk-size <N>` | `1500` | Max chars per model inference pass — lower values reduce peak VRAM |

All flags have `ANON_PROXY_*` env-var equivalents (`ANON_PROXY_HOST`, `ANON_PROXY_PORT`, `ANON_PROXY_UPSTREAM`, `ANON_PROXY_BACKEND`, `ANON_PROXY_DEBUG=1`, `ANON_PROXY_PATTERNS`, `ANON_PROXY_MERGE_GAP`, `ANON_PROXY_CHUNK_SIZE`).

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

With `--debug`, each request prints a compact diff to stderr:
```
==== POST /v1/messages | model=claude-opus-4-7 | 3 msg ====
[store +2]
  <PERSON_1>  ←  'Alice Smith'
  <EMAIL_1>   ←  'alice@company.com'
[masked]
  user[2] text: 'Fix the bug reported by Alice Smith (alice@company.com)…'
              → 'Fix the bug reported by <PERSON_1> (<EMAIL_1>)…'
[unmasked stream] 'I'll fix the bug for <PERSON_1>…' → 'I'll fix the bug for Alice Smith…'
```

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

- **Usability** : Add OpenAI API adapter for broader client compatibility, then expand to other providers.
- **Quality assurance** : Enhance PII detection quality tracking and add comprehensive unit/integration tests with benchmarking.
- **Observability** : Implement structured logging and telemetry for monitoring proxy performance and PII masking metrics.
- **Persistence** : Optionally persist PII mappings to disk so placeholder consistency survives server restarts.
- **Dev infrastructure** : Set up CI, contribution guidelines, and project templates to streamline community development.
