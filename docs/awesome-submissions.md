# Awesome-list submission packet

Pre-formatted entries for getting anon-proxy listed on curated "awesome" lists.
Each section is one target list. Copy the entry text, open a PR against the
list's repo following its CONTRIBUTING guide, and link back to anon-proxy.

## How to submit (general)

1. Fork the target list's repo.
2. Find the right section (privacy, LLM tooling, proxies, etc.).
3. Insert the entry alphabetically *within* that section unless the list
   says otherwise.
4. Follow the list's exact link format — most use
   `- [name](url) - description.` but capitalisation/punctuation varies.
5. Run any linter the list ships (`awesome-lint`, custom CI) before
   opening the PR.
6. PR title: typically `Add anon-proxy` or `Add anon-proxy to <section>`.

## Target lists

### sindresorhus/awesome (the meta-list)

Don't submit here — entries must themselves be awesome lists, not projects.

### awesome-llm-apps / awesome-llm

Repos to check (verify each is still active before submitting; some have
gone stale):
- https://github.com/Shubhamsaboo/awesome-llm-apps
- https://github.com/Hannibal046/Awesome-LLM
- https://github.com/horseee/Awesome-Efficient-LLM

**Section to target:** "Tools" or "Infrastructure" or "Privacy" — whichever
exists. If only "Applications" exists, look for a "Tools" subsection.

**Entry text:**

```markdown
- [anon-proxy](https://github.com/KevinXuxuxu/anon_proxy) - Local PII masking proxy for LLM APIs (Anthropic, OpenAI). Detects names, emails, and phones with the openai/privacy-filter model and rewrites them to stable placeholders before requests leave your machine. Works with Claude Code and the OpenAI SDK.
```

### awesome-privacy

Target: https://github.com/Lissy93/awesome-privacy

**Section to target:** "Developer tools" or a "PII / Data protection" section
if one exists.

**Entry text** (this list uses a structured YAML-ish front-matter, check the
CONTRIBUTING guide before submitting):

```markdown
- name: anon-proxy
  url: https://github.com/KevinXuxuxu/anon_proxy
  description: Local PII masking proxy for LLM APIs. Detects PII with the openai/privacy-filter model and rewrites to stable placeholders before requests leave the device.
  icon: shield
```

If the file is plain Markdown after all, fall back to the standard one-line
format above.

### awesome-mlops / awesome-ml-tools

- https://github.com/kelvins/awesome-mlops

**Section:** "Privacy" or "Inference / Serving."

**Entry text:** same one-line format as awesome-llm above.

### awesome-claude

If a curated list specifically for Claude/Anthropic exists, it's a strong fit.
Check:
- https://github.com/eastlondoner/awesome-claude-code (if it exists)
- Search GitHub: `awesome claude` topic + recent commits.

If none active: skip.

### Hacker News "Show HN"

Not a list, but high-leverage one-shot promotion. Post format:

> **Show HN: anon-proxy – Local PII masking proxy for LLM APIs**
>
> https://github.com/KevinXuxuxu/anon_proxy
>
> Hi HN — anon-proxy is a small Starlette/uvicorn proxy that sits between
> your app and Anthropic/OpenAI. Before each request goes upstream, it runs
> the openai/privacy-filter NER model locally and rewrites detected PII
> (names, emails, phones, addresses) to stable placeholders like
> `<PERSON_1>`. The model's response uses the same placeholders, which
> the proxy unmasks before returning to your client — so the upstream
> provider never sees the raw values, but the application logic still works.
>
> Built it because I wanted to use Claude Code and the OpenAI SDK on
> internal data without sending the raw text to a vendor. Open to
> feedback on the threat model, the placeholder scheme, and the missing
> features (full streaming, more providers).

Best to post on a Tuesday-Thursday morning Pacific time.

### Reddit / r/MachineLearning, r/LocalLLaMA, r/Privacy

Format is more conversational — share what you built, what problem it
solves, and ask for feedback. Read the subreddit's self-promotion rules
before posting; some require a 9:1 contribution-to-promotion ratio.

## Submission tracker

Keep a row per submission so we don't double-submit:

| Date | List / venue | Entry URL (PR / post) | Status |
|------|--------------|------------------------|--------|
| YYYY-MM-DD | awesome-llm-apps | https://… | open / merged / rejected |
