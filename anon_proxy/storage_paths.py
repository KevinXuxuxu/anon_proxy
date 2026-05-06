"""OS-aware default paths for anon-proxy telemetry, with sync-root warnings.

The default for raw telemetry moves out of `~/.anon-proxy/` to a directory
that is, by default, not synced to consumer cloud storage:

- macOS: ~/Library/Application Support/anon-proxy/
- Linux: $XDG_DATA_HOME/anon-proxy/  (typically ~/.local/share/anon-proxy/)

`is_under_sync_root()` detects when a user-supplied path lives under a
known sync root so `server.py` can warn loudly. Encryption protects the
bytes, but a synced encrypted blob still leaves a permanent copy on
provider infrastructure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncRoot:
    name: str
    relative: str  # path relative to $HOME


SYNC_ROOTS: tuple[SyncRoot, ...] = (
    SyncRoot("Dropbox", "Dropbox"),
    SyncRoot("iCloud Drive", "Library/Mobile Documents/com~apple~CloudDocs"),
    SyncRoot("iCloud Drive", "iCloud Drive"),
    SyncRoot("OneDrive", "OneDrive"),
    SyncRoot("Sync", "Sync"),
    SyncRoot("Google Drive", "Google Drive"),
)


def default_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path(os.environ["HOME"]) / "Library" / "Application Support" / "anon-proxy"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "anon-proxy"
    return Path(os.environ["HOME"]) / ".local" / "share" / "anon-proxy"


def is_under_sync_root(path: Path) -> str | None:
    """Return the sync-root name if `path` lives under one, else None."""
    home = Path(os.environ["HOME"]).resolve()
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    try:
        rel = resolved.relative_to(home)
    except ValueError:
        return None
    for root in SYNC_ROOTS:
        root_parts = Path(root.relative).parts
        if rel.parts[: len(root_parts)] == root_parts:
            return root.name
    return None


def secure_create_dir(path: Path) -> None:
    """Create dir with 0700 perms. Idempotent."""
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def secure_create_file(path: Path) -> None:
    """Create empty file with 0600 perms if it doesn't exist."""
    secure_create_dir(path.parent)
    if not path.exists():
        path.touch()
    path.chmod(0o600)


def exclude_from_time_machine(path: Path) -> bool:
    """Best-effort `tmutil addexclusion`. Returns True on success, False otherwise.

    Caller should log the False case so the user knows backups may include the
    encrypted blob.
    """
    if sys.platform != "darwin":
        return False
    try:
        subprocess.run(
            ["tmutil", "addexclusion", str(path)],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
