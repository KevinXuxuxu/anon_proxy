"""Static analysis of the anon-proxy-triage skill: assert every CLI invocation
the skill is documented to make includes --metadata-only or appears in the
explicit Forbidden actions section.

This is the CI safety net for the skill's trust boundary. If a future edit to
SKILL.md sneaks in a `triage` invocation without --metadata-only, this test
fails before the change can ship."""

import re
from pathlib import Path

import pytest


SKILL_PATH = Path(".claude/skills/anon-proxy-triage/SKILL.md")
TRIAGE_CMD_RE = re.compile(r"`(anon-proxy telemetry triage[^`]*?)`")
ALLOWED_INTERACTIVE = re.compile(r"\banon-proxy telemetry triage --days \d+\b")


def test_skill_file_exists():
    assert SKILL_PATH.exists(), f"skill file missing at {SKILL_PATH}"


def test_skill_has_required_sections():
    text = SKILL_PATH.read_text()
    assert "## Allowed CLI invocations" in text
    assert "## Forbidden actions" in text
    assert "## Workflow" in text


def test_every_triage_invocation_has_metadata_only_or_is_in_forbidden_or_interactive():
    """Every fenced `anon-proxy telemetry triage ...` snippet in the skill must
    EITHER include --metadata-only, OR be the documented interactive escape
    (used by the human in their terminal, not by the skill), OR live in the
    Forbidden actions section as an example of what NOT to call."""
    text = SKILL_PATH.read_text()
    forbidden_section = text.split("## Forbidden actions")[1].split("\n## ")[0]
    workflow_section = text.split("## Workflow")[1].split("\n## ")[0] if "## Workflow" in text else ""

    bad = []
    for match in TRIAGE_CMD_RE.finditer(text):
        cmd = match.group(1)
        if "--metadata-only" in cmd:
            continue
        # Allow the interactive form mentioned to the USER (not invoked by the skill)
        if cmd in workflow_section and ALLOWED_INTERACTIVE.fullmatch(cmd):
            continue
        if cmd in forbidden_section:
            continue
        bad.append(cmd)

    assert not bad, (
        "These triage invocations are missing --metadata-only and aren't in "
        f"the Forbidden actions / interactive-escape allowlist: {bad}"
    )


def test_skill_explicitly_forbids_show_and_with_text():
    text = SKILL_PATH.read_text()
    forbidden = text.split("## Forbidden actions")[1].split("\n## ")[0]
    assert "corpus show" in forbidden, "skill must explicitly forbid `corpus show` (decrypts)"
    assert "--with-text" in forbidden, "skill must explicitly forbid `--with-text` (decrypts)"


def test_skill_frontmatter_present():
    """Claude Code skills require YAML frontmatter with name + description."""
    text = SKILL_PATH.read_text()
    assert text.startswith("---\n"), "skill must start with YAML frontmatter"
    fm_end = text.index("---", 4)
    fm = text[4:fm_end]
    assert "name:" in fm
    assert "description:" in fm
