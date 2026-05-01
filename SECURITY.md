# Security policy

anon-proxy is a privacy tool. The whole point is preventing PII from leaking to
upstream LLM APIs. Bugs that defeat that goal are the most important class of
issue this project has.

## Reporting a vulnerability

**Do not file a public GitHub issue for security bugs.** A public issue with a
working exploit invites the same data leaks anon-proxy is supposed to prevent.

Instead, use GitHub's [Private Vulnerability Reporting][pvr] to send the
maintainers a private advisory:

1. Go to the [Security tab](https://github.com/KevinXuxuxu/anon_proxy/security)
   of this repository.
2. Click **Report a vulnerability** (or open
   <https://github.com/KevinXuxuxu/anon_proxy/security/advisories/new> directly).
3. Fill in the advisory form.

Please include:

- A short description of what the bug allows.
- Steps to reproduce, ideally with a minimal input (a prompt, a tool call, etc.)
  that demonstrates PII leaking past the masker.
- The detector configuration you were running (default model, any `--patterns`
  or `--merge-gap-file` overrides, `--backend` setting).
- Your view on severity (does it leak in normal use, or only with a contrived
  setup?).

You should expect an acknowledgement within a few days. Once a fix is ready
and shipped, the advisory can be published with credit.

[pvr]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

## Threat model

### In scope

Scope is limited to **the providers anon-proxy ships an adapter for** —
currently the Anthropic Messages API (`/anthropic`) and the OpenAI
Chat Completions API (`/openai`). Custom upstreams added via
`--extra-upstream` are in scope only when paired with one of the supported
adapter types (`adapter=anthropic` or `adapter=openai`); other routings
will pass raw bytes through.

- **Outbound request bodies on supported adapters** — anything that goes
  from your client through a supported adapter to the upstream API. Names,
  emails, phone numbers, addresses, and other configured PII categories
  should be replaced with stable placeholders before the bytes leave the
  proxy process.
- **Inbound response bodies on supported adapters** — placeholders the
  model emits should be rewritten to the original values before the
  response reaches your client, so the client sees a coherent conversation.
- **Multi-turn coherence** — the same input value should map to the same
  placeholder on every turn within a session, so the model can reason about
  the same entity over time.

### Out of scope

These are not bugs anon-proxy claims to defend against:

- **A malicious local user.** anon-proxy runs on your machine and trusts the
  user running it. Anyone with shell access to the host can read the in-memory
  store, inspect uvicorn logs, or just set `--debug`.
- **Side channels.** Request length, latency, and traffic shape are not
  obscured. An upstream provider observing many proxied requests can still
  infer aggregate properties.
- **The system prompt and tool definitions.** These are passed through
  unmasked because they typically contain static instructions and schemas, not
  user PII. If your system prompt contains PII, treat that as a misuse — put
  the PII in user/assistant messages where the masker runs.
- **Extended-thinking blocks.** Anthropic's extended-thinking blocks carry
  cryptographic signatures that break if the contents are rewritten, so they
  are passed through unchanged. Don't put PII in fields a model is going to
  emit as signed thinking.
- **The detector model itself.** anon-proxy uses the
  [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model.
  False negatives in that model are detector quality bugs, not anon-proxy
  vulnerabilities — but if you find a class of input that systematically
  bypasses masking, please report it; we may be able to mitigate it via
  chunking, regex augmentation, or merge-gap tuning.

### Known limitations

These are documented gaps rather than secrets. Please do *not* file private
security reports for these — they are open trade-offs:

- **Clue-less PII can be missed.** Bare phone numbers, isolated tokens, or
  out-of-context identifiers may evade the ML detector. Promote regex
  detectors via `--patterns` to close gaps you observe.
- **Span fragmentation.** A single entity may be split into adjacent spans
  ("Jean" + "Luc"). Adjust `--merge-gap-file` per label.
- **Chunk boundaries.** Long inputs are chunked at `--chunk-size`; PII
  straddling a boundary may lose context. Raise `--chunk-size` if you have
  the VRAM.
- **Tool results that depend on literal values.** If a tool consumes a literal
  email or ID and round-trips it back, the masked placeholder may break the
  tool's contract. Configure `--patterns` carefully or skip masking for the
  affected tool.

## Operational guidance

If you run anon-proxy on multi-user hardware or expose it on a LAN
(`--host 0.0.0.0`), you are widening the trust boundary beyond a single user.
The proxy has no authentication of its own — it forwards client auth headers
unchanged. Put your own auth in front of it (firewall, mTLS, an
authenticating reverse proxy) before exposing it beyond `127.0.0.1`.

When in doubt, run with `--debug` once on representative traffic and read the
diffs. Anything you see leaving the proxy unmasked, the upstream provider sees
too.
