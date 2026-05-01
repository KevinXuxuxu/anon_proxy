<img width="108" height="112" alt="anon" src="https://github.com/user-attachments/assets/6609f7ff-3e0b-458d-ac20-2f1b0b95ae62" />

# anon-proxy

**Use Claude Code, ChatGPT, and other LLM APIs on sensitive data without sending raw PII to the cloud.** A local privacy proxy that masks personal information *before* requests leave your device and unmasks it in responses. The [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model runs entirely on your machine — names, emails, phone numbers, and addresses never reach Anthropic, OpenAI, or any other upstream API.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Works with Claude Code](https://img.shields.io/badge/works%20with-Claude%20Code-orange)

```
your client  →  anon-proxy (mask/unmask)  →  api.anthropic.com | api.openai.com | ...
```

---

## Who is this for?

- **Engineers in regulated industries** (healthcare, legal, finance) whose employer policy or compliance regime (HIPAA, GDPR, SOC 2) blocks raw customer data from being sent to third-party LLM APIs.
- **Developers using Claude Code or OpenAI SDKs on production data** — debugging customer support tickets, summarizing user emails, analyzing logs that contain real names and identifiers.
- **Privacy-conscious users** who want LLM productivity without the data exhaust of pasting personal information into a cloud model.
- **Self-hosters** building on top of LLM APIs who need a redaction layer between their app and the provider.

If you've ever caught yourself manually find-and-replacing real names in a prompt, this is for you.

---

## Why not just use X?

| Tool | What it does | Why it's not the same |
|---|---|---|
| **Microsoft Presidio** | Regex + spaCy NER for PII detection | Library, not a proxy. You still have to wire it into every LLM call yourself. No stable token mapping across turns. |
| **AWS Comprehend / GCP DLP** | Cloud-based PII detection APIs | Sends your data to *another* cloud provider. Defeats the purpose if your goal is "nothing leaves the box." |
| **LiteLLM proxy** | Multi-provider LLM routing | Doesn't redact. Solves a different problem (routing/cost) entirely. |
| **Prompting "please don't log my data"** | 🙏 | Not a security model. |
| **anon-proxy** | Local ML detector + transparent proxy with stable per-session placeholders | Drop-in `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` swap. No code changes in the client. PII gets the same placeholder every turn so the model stays coherent. |

---

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

**What is NOT masked:** the system prompt (tool schemas and static instructions), tool definitions, and extended-thinking blocks (signatures would break). See [`SECURITY.md`](SECURITY.md) for the full threat model and known limitations.

**How it works:** PII spans get stable placeholder tokens (`<PERSON_1>`, `<EMAIL_1>`, `<ADDRESS_1>`, …) stored in a per-session dictionary. The same value always maps to the same token across turns so the model stays coherent. Responses are unmasked before reaching your client.

---

## Configuration files

| File | Purpose |
|---|---|
| `patterns.json` | Extra regex patterns for PII the ML model misses (SSNs, IPs, internal IDs) |
| `merge_gap.json` | Per-label chars allowed inside a gap when merging adjacent spans (e.g. hyphen for `PERSON` so "Jean-Luc" → one token) |

Copy from the `.example` files to get started.

---

## FAQ

### How is this different from Microsoft Presidio?

Presidio is a Python library for PII detection — you call it from your code. anon-proxy is a transparent network proxy: you point your existing LLM client at it via `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL` and it intercepts every request. It also maintains a stable token↔value mapping across turns, so the model sees the same `<PERSON_1>` on turn 5 that it saw on turn 1 (Presidio doesn't do this — that's an application-layer concern it leaves to you).

### Does this work with Claude Code?

Yes. That's the primary supported client. Set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic` and run `claude` normally. See [Using with Claude Code](#using-with-claude-code).

### Does this work with OpenAI / ChatGPT SDK clients?

Yes — point your OpenAI client at `http://127.0.0.1:8080/openai`. See [Using with OpenAI SDK](#using-with-openai-sdk).

### Will the LLM still understand my prompt after PII is replaced with placeholders?

Generally yes. Modern LLMs treat `<PERSON_1>` and `<EMAIL_1>` as opaque variable names and reason about relationships between them. The stable mapping (same person → same token across turns) is what makes multi-turn conversations work — without it, "tell <PERSON_1> about the meeting" on turn 2 would refer to a different person than turn 1's `<PERSON_1>`.

### Does masking break tool calls?

Tool inputs and tool results are masked/unmasked just like message text, so most tools work unchanged. Caveat: if a tool's behavior depends on the literal value of a PII string (e.g. a database lookup that takes an email), you'll want to register that field in `--patterns` carefully — or skip masking for that tool. Extended-thinking blocks are passed through unmasked because their cryptographic signatures would break.

### What PII does it detect?

Out of the box: persons, emails, phone numbers, addresses, organizations, dates of birth, government IDs, and other categories from the openai/privacy-filter model. Add your own (SSN formats, internal employee IDs, project codenames) via `--patterns`.

### What's the performance overhead?

We don't have published benchmark numbers yet — latency measurement is on the roadmap.

### Is the threat model documented?

Yes — see [`SECURITY.md`](SECURITY.md). It covers what's in scope (request bodies leaving your machine through the supported adapters), what's out of scope (a malicious local user, side channels, the system prompt itself), and known false-negative modes.

### How do I report a vulnerability?

See [`SECURITY.md`](SECURITY.md). For a privacy tool, *quietly* is usually better than a public issue — please email the maintainer first.

---

## Next steps / roadmap

- **Quality assurance** : Enhance PII detection quality tracking and add comprehensive unit/integration tests with benchmarking.
- **Observability** : Implement structured logging and telemetry for monitoring proxy performance and PII masking metrics.
- **Persistence** : Optionally persist PII mappings to disk so placeholder consistency survives server restarts.
- **Usability** : Now supporting Anthropic and OpenAI APIs, but need more compatibility testing and expand to other potential providers.
- **Dev infrastructure** : Set up CI, contribution guidelines, and project templates to streamline community development.

---

## License & security

- Licensed under the [MIT License](LICENSE).
- For security disclosures and the threat model, see [`SECURITY.md`](SECURITY.md).
- Issues and PRs welcome — this is a young project and feedback is the fastest way to improve it.
