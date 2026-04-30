---
name: anon-proxy-triage
description: Use when the user wants to review anon_proxy telemetry — triaging false positives / negatives, building a labeled benchmark corpus, mining slipped PII for new regexes, or summarizing detector trends. The skill orchestrates the local `anon-proxy telemetry` CLI and reasons about structural metadata only — it never decrypts or sends raw PII to the LLM API.
---

# anon-proxy telemetry triage

This skill helps the user maintain anon_proxy detector quality over time without ever sending raw PII to the upstream LLM. **Trust boundary: every CLI call MUST include `--metadata-only`.** Anything that would surface raw entity text or context windows happens in the user's local terminal, not in this skill's reasoning.

## Allowed CLI invocations

These are the only commands this skill may run. All include `--metadata-only` where applicable.

| Purpose | Command |
|---|---|
| Read trend metrics | `anon-proxy telemetry metrics --json` |
| Read structural metadata for triage | `anon-proxy telemetry triage --json --metadata-only --days N` |
| Read labeled corpus stats | `anon-proxy telemetry corpus list` |
| Mechanical regex synthesis | `anon-proxy telemetry suggest-regex --from-corpus` |
| Run offline eval against the labeled corpus | `uv run python -m anon_proxy.eval --corpus <path>` |

## Forbidden actions

- **Never call** `anon-proxy telemetry triage` without `--metadata-only`.
- **Never call** `anon-proxy telemetry corpus show` (decrypts). Instead, instruct the user to run it themselves.
- **Never call** `anon-proxy telemetry-report --with-text`.
- **Never** read `telemetry-raw.jsonl` or `corpus.jsonl` directly with shell tools — those files contain encrypted PII that the user might not want decrypted. Always go through the CLI.

## Workflow

When the user asks "review my anon-proxy telemetry":

1. Run `anon-proxy telemetry metrics --json --since <30 days ago>`. Summarize the trend.
2. Run `anon-proxy telemetry triage --json --metadata-only --days 7`. Identify clusters by `signature` (added by the CLI for non-metadata-only mode — but here we get raw label/source/score patterns only, since `--metadata-only` strips signatures).
3. Surface "what to look at this week" by structural pattern. Tell the user the count and pattern, NOT the content. E.g., "5 PHONE candidates flagged by baseline regex but missed by ML, all with country code +44."
4. Suggest the user run `anon-proxy telemetry triage --days 7` (no `--metadata-only`, no `--json` — interactive) to make adjudication calls in their terminal.
5. After they label, run `anon-proxy telemetry suggest-regex --from-corpus`. Take its mechanical output and propose any improvements based on signature patterns.
6. Run `uv run python -m anon_proxy.eval --corpus <path>` to measure P/R/F1 before and after the proposed regexes.

## What the skill is NOT for

- Deciding "is this PII?" — that's a human judgment call in the local CLI.
- Reading raw entity text or context windows.
- Bulk decryption of any kind.
