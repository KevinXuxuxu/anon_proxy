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
| `--telemetry` | off | Log one JSON record per masked request to a local JSONL file (no PII content) |
| `--telemetry-path <file>` | `~/.anon-proxy/telemetry.jsonl` | Override telemetry log path |

All flags have `ANON_PROXY_*` env-var equivalents (`ANON_PROXY_HOST`, `ANON_PROXY_PORT`, `ANON_PROXY_DEBUG=1`, `ANON_PROXY_PATTERNS`, `ANON_PROXY_MERGE_GAP`, `ANON_PROXY_CHUNK_SIZE`, `ANON_PROXY_BACKEND`, `ANON_PROXY_TELEMETRY=1`, `ANON_PROXY_TELEMETRY_PATH`).

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

## Pipeline architecture

`Masker.mask(text)` runs five named stages. The first three are detection
passes; the fourth resolves overlaps; the fifth replaces. Pass 3 only runs
when telemetry is enabled and never affects the masked output.

```
text
 │
 ├─[Pass 1] detect_ml ─────────────► ml_spans          (chunked ML inference)
 ├─[Pass 2] detect_user ──────────► user_spans        (user RegexDetectors)
 ├─[Pass 3] detect_baseline ──────► baseline_spans    (observer-only; --telemetry)
 │
 ▼
[Pass 4] resolve  (greedy: longer wins; score breaks ties)
 │
 ▼
[Pass 5] replace  (right-to-left token substitution)
```

| Concern | Lives in |
|---|---|
| Chunking (whitespace-bounded splitter) | `anon_proxy/privacy_filter.py:_split_chunks` (Pass 1 only) |
| Same-label cross-chunk merge | `anon_proxy/privacy_filter.py:_merge_adjacent_entities` (inside Pass 1) |
| User regex detectors | `anon_proxy/regex_detector.py` (Pass 2) |
| Baseline regex (observer) | `anon_proxy/telemetry_patterns.json` (Pass 3) |
| Range interaction policy | `anon_proxy/pipeline.py:GreedyLongerWins` (Pass 4) |

User regex (Pass 2) and baseline regex (Pass 3) operate on the full text;
chunking is purely a Pass 1 concern.

When two spans overlap, the longer one wins; ties are broken by score.
Touching spans (`prev.end == next.start`) are *not* overlapping. Regex
detectors emit `score=1.0`, so on a length tie regex beats ML — most
commonly relevant when a regex stitches two ML fragments.

## Offline eval

To get real precision/recall numbers against a labeled corpus:

```bash
# Bundled synthetic corpus (200 examples, seedable, zero real PII):
uv run python -m anon_proxy.eval

# OPF smoke test (5 samples from openai/privacy-filter, Apache 2.0):
uv run python -m anon_proxy.eval --corpus anon_proxy/eval_corpus/opf_samples.jsonl

# Bring your own labeled JSONL: {"text": "...", "spans": [{"label": ..., "start": ..., "end": ...}]}
uv run python -m anon_proxy.eval --corpus path/to/yours.jsonl
```

Output: per-label P/R/F1/n for each requested detector. Note: the synthetic
corpus's recall numbers reflect its template distribution, not your
production traffic — see `--telemetry` for traffic-level evidence.

---

## Telemetry (optional)

The ML detector can silently miss PII — especially clue-less values like bare phone numbers or pasted tokens. To measure how often this happens *on your actual traffic*, run the proxy with `--telemetry`:

```bash
uv run python -m anon_proxy.server --telemetry
```

One JSON record **per API request** is appended to `~/.anon-proxy/telemetry.jsonl`. The record aggregates every `Masker.mask()` call the Anthropic adapter made while handling that `POST /v1/messages` — all text blocks, all tool-use inputs, all tool results — into a single line. Records contain labels, lengths, and boundary flags only — never the original PII value, never a slice of the request text. A built-in conservative regex set (email / phone / SSN / IPv4 / IPv6) is run alongside the ML detector and flags any span the model missed.

After a day or two of normal use, summarize the log:

```bash
uv run python -m anon_proxy.telemetry_report
```

```
2,147 API requests
  avg request:   1,843 chars, 1.40 chunks (312 multi-chunk, 15%)

ML detector: 8,412 spans
  private_person  4,281
  private_email   2,104
  ...

Baseline regex caught but detectors missed: 47 spans
  PHONE_NANP   41
  EMAIL         3
  SSN          18

Miss characterization:
  no detector span within 50ch (isolated):   42 ( 89%)
  nearest detector span same label:           2 (  4%)  (suggests fragmentation)
  within 50ch of a real chunk boundary:       3 (  6%)  (suggests chunking cost)
```

Interpretation:
- **Mostly "isolated" misses** → clue-less PII is the dominant failure mode; consider promoting the regex detectors to real maskers via `--patterns`.
- **Many "same-label nearby" misses** → the model fragmented a span and merge-gap didn't stitch it; adjust `--merge-gap-file`.
- **Many "near real chunk boundary" misses** → chunk boundaries are stripping context; raise `--chunk-size`.

### Latency (`latency_ms` field)

When the proxy serves a request, each v2 record also carries:

```json
"latency_ms": {"mask": 3, "upstream": 412, "unmask": 2, "total": 420}
```

- `mask` — milliseconds spent in `Masker.mask` for the request (Pass 1–4 of the
  pipeline plus token replacement).
- `upstream` — time from request-sent to upstream-response-complete. For
  streaming responses this includes the full stream duration.
- `unmask` — time spent unmasking. Non-streaming: a single pass. Streaming:
  cumulative across all chunks (microseconds summed inside `transform_stream`,
  truncated to ms on commit).
- `total` — end-to-end proxy latency (request received → response complete).

For streaming requests `mask + upstream + unmask` will not equal `total`
because mask runs before upstream and unmask is interleaved with upstream.

`python -m anon_proxy.telemetry_report` prints per-phase p50/p95 across all
logged requests.

If you want stricter regex coverage (e.g. international phone formats), pass your own via `--patterns` — user regexes count as "caught" in telemetry, so the log shows you what's *still* leaking after every configured layer runs.

## Next steps / roadmap

- **Quality assurance** : Enhance PII detection quality tracking and add comprehensive unit/integration tests with benchmarking.
- **Observability** : Implement structured logging and telemetry for monitoring proxy performance and PII masking metrics.
- **Persistence** : Optionally persist PII mappings to disk so placeholder consistency survives server restarts.
- **Usability** : Now supporting Anthropic and OpenAI APIs, but need more compatibility testing and expand to other potential providers.
- **Dev infrastructure** : Set up CI, contribution guidelines, and project templates to streamline community development.
