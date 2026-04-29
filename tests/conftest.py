"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_keyring(monkeypatch):
    """In-memory keyring substitute. Use in any test that touches anon_proxy.crypto."""
    store: dict[tuple[str, str], str] = {}

    class _FakeKeyring:
        @staticmethod
        def get_password(service, user):
            return store.get((service, user))

        @staticmethod
        def set_password(service, user, value):
            store[(service, user)] = value

        @staticmethod
        def delete_password(service, user):
            store.pop((service, user), None)

    import anon_proxy.crypto as crypto_mod
    monkeypatch.setattr(crypto_mod, "keyring", _FakeKeyring)
    return store
