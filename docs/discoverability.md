# GitHub repo discoverability checklist

This is a one-time checklist for the upstream maintainer. Each item is
something GitHub indexes for search, or that AI agents and curators see when
they look at the repo.

## 1. Repo description (About section)

Set on https://github.com/KevinXuxuxu/anon_proxy → ⚙️ next to "About":

> Local-first PII masking proxy for LLM APIs. Detects names, emails, phones, addresses with openai/privacy-filter and rewrites them to stable placeholders before bytes leave your machine. Works with Claude Code, OpenAI SDK, and any HTTP-compatible client.

(255-char limit. The current text is the GitHub default or empty.)

## 2. Website link

In the same About panel, set "Website" to the README URL or a project page:
`https://github.com/KevinXuxuxu/anon_proxy#readme`

## 3. Topics (high-impact for GitHub search)

Add these topics in the About panel. GitHub allows up to 20.

**Core (must-have):**
- `privacy`
- `pii`
- `pii-detection`
- `llm`
- `proxy`
- `data-protection`

**Provider/client targeting (helps users find you):**
- `anthropic`
- `claude`
- `openai`
- `chatgpt`
- `claude-code`

**Stack:**
- `python`
- `transformers`
- `huggingface`

**Use case:**
- `redaction`
- `masking`
- `data-anonymization`
- `gdpr`

## 4. Social preview image

GitHub Settings → "Social preview" → upload a 1280×640 PNG/JPG.
This is what shows in Twitter/Slack/Discord cards when the repo is shared.

Suggested content for the preview image:
- Top half: project name + tagline ("PII masking proxy for LLM APIs").
- Bottom half: a small diagram or terminal screenshot showing
  `<PERSON_1>` placeholders replacing real names in a request.

If you don't have design tooling, GitHub's auto-generated preview (repo name
on a colored background) is acceptable but generic. Even a quick Figma export
beats it.

## 5. Pinned repos on the maintainer profile

If you maintain multiple repos, pin anon-proxy on your GitHub profile so
visitors see it first.

## 6. README badge audit

Confirm the badges at the top of `README.md` actually link somewhere useful:
- License badge → `LICENSE` file
- Python version badge → relevant docs
- "Works with Claude Code" badge → README anchor or Anthropic docs

## 7. CODEOWNERS (optional but signals maintenance)

Even a one-line `.github/CODEOWNERS` makes it clear who reviews PRs:

```
* @KevinXuxuxu
```

## 8. Issue/PR templates (optional)

If issue volume grows, add `.github/ISSUE_TEMPLATE/bug_report.md` and
`.github/ISSUE_TEMPLATE/feature_request.md`. For a young project this is
overkill — wait until repeated low-quality issues justify the friction.
