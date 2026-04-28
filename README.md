<img width="108" height="112" alt="anon" src="https://github.com/user-attachments/assets/6609f7ff-3e0b-458d-ac20-2f1b0b95ae62" />

# anon-proxy

An LLM API proxy that masks PII before requests leave your device and unmasks it in responses. The [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model runs **locally** — raw PII never reaches the upstream API.

```
your client  →  anon-proxy (mask/unmask)  →  api.anthropic.com | api.openai.com | ...
```

## Multi-provider support

The proxy uses **sub-routing** to support multiple API providers:

```
/{provider}/{api-path}  →  {provider-base-url}/{api-path}
```

Examples:
- `/anthropic/v1/messages` → `https://api.anthropic.com/v1/messages`
- `/openai/v1/chat/completions` → `https://api.openai.com/v1/chat/completions`
- `/zai/v1/messages` → `https://api.z.ai/api/anthropic/v1/messages`

Built-in providers: `anthropic`, `openai`, `zai`. Add custom providers with `--extra-upstream`.

---

## Quick demo

```bash
# test the PII detector interactively
uv run python test_filter.py "Alice Smith called from 555-867-5309, email alice@company.com"
```
```
[private_person:Alice] [private_person:Smith] called from [private_phone:555-867-5309], email [private_email:alice@company.com]

  private_person 'Alice'                        score=1.000  offset=0-5
  private_person 'Smith'                        score=1.000  offset=6-11
  private_phone '555-867-5309'                 score=1.000  offset=24-36
  private_email 'alice@company.com'            score=1.000  offset=44-61
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
| `--backend` | `auto` | PII detection backend (`auto`, `cpu`, `mps`, `mlx`) |
| `--extra-upstream` | — | Add custom provider: `name=url[;adapter=anthropic\|openai][;path_prefix=/path]` |
| `--debug` | off | Log new store entries and masked/unmasked diffs to stderr |
| `--patterns <file>` | — | JSON file of extra regex detectors: `{"LABEL": "regex", ...}` |
| `--merge-gap-file <file>` | — | JSON file overriding per-label adjacency merge chars (see `merge_gap.json.example`) |
| `--chunk-size <N>` | `1500` | Max chars per model inference pass — lower values reduce peak VRAM |

**Add a custom provider:**
```bash
uv run python -m anon_proxy.server \
  --extra-upstream "myprovider=https://api.example.com;adapter=anthropic"
```

Then use: `base_url=http://127.0.0.1:8080/myprovider`

**With all config files:**
```bash
uv run python -m anon_proxy.server \
  --patterns patterns.json \
  --merge-gap-file merge_gap.json \
  --backend mps \
  --debug
```

## Testing with the proxy

Test the PII masking through the proxy using `test_mask.py`:

```bash
# Start the proxy
uv run python -m anon_proxy.server --debug

# In another terminal, test with Anthropic (--no-mask means proxy handles masking)
ANTHROPIC_API_KEY=sk-ant-... \
ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic \
uv run python test_mask.py --provider anthropic --no-mask

# Or test with OpenAI
OPENAI_API_KEY=sk-... \
OPENAI_BASE_URL=http://127.0.0.1:8080/openai \
uv run python test_mask.py --provider openai --no-mask
```

---

## Using with Claude Code

Point Claude Code at the proxy (note the provider prefix in the URL):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic claude
```

Or set it permanently in `~/.zshrc` / `~/.bashrc`:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic
```

No other changes — the proxy forwards your auth headers unchanged.

## Using with OpenAI SDK

For OpenAI-compatible clients, use the `/openai` provider path:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/openai python your_openai_app.py
```

Or export it permanently:
```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/openai
```

## Debug output

With `--debug`, each request prints a compact diff to stderr:
```
==== anthropic /v1/messages | model=claude-opus-4-7 | 3 msg ====
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

- **Quality assurance** : Enhance PII detection quality tracking and add comprehensive unit/integration tests with benchmarking.
- **Observability** : Implement structured logging and telemetry for monitoring proxy performance and PII masking metrics.
- **Persistence** : Optionally persist PII mappings to disk so placeholder consistency survives server restarts.
- **Usability** : Now supporting Anthropic and OpenAI APIs, but need more compatibility testing and expand to other potential providers.
- **Dev infrastructure** : Set up CI, contribution guidelines, and project templates to streamline community development.
