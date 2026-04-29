import os
import sys
from pathlib import Path

import pytest

from anon_proxy.storage_paths import (
    default_data_dir,
    is_under_sync_root,
    SYNC_ROOTS,
)


def test_default_data_dir_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_data_dir() == tmp_path / "Library" / "Application Support" / "anon-proxy"


def test_default_data_dir_linux_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_data_dir() == tmp_path / "xdg" / "anon-proxy"


def test_default_data_dir_linux_no_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_data_dir() == tmp_path / ".local" / "share" / "anon-proxy"


def test_is_under_sync_root_dropbox(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / "Dropbox" / "anon-proxy" / "telemetry-raw.jsonl"
    assert is_under_sync_root(p) == "Dropbox"


def test_is_under_sync_root_clean(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / "Library" / "Application Support" / "anon-proxy" / "f.jsonl"
    assert is_under_sync_root(p) is None


def test_sync_roots_contains_known_services():
    names = {r.name for r in SYNC_ROOTS}
    assert {"Dropbox", "iCloud Drive", "OneDrive", "Sync"} <= names
