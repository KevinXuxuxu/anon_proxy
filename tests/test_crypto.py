import base64

import pytest

from anon_proxy.crypto import (
    encrypt_field,
    decrypt_field,
    generate_key,
    SCHEME_PREFIX,
    DecryptionError,
)


def test_encrypt_round_trip():
    key = generate_key()
    ct = encrypt_field("Alice Smith", key)
    assert ct.startswith(SCHEME_PREFIX)
    assert decrypt_field(ct, key) == "Alice Smith"


def test_encrypt_unicode_round_trip():
    key = generate_key()
    plaintext = "Olivia Müller — naïve"
    assert decrypt_field(encrypt_field(plaintext, key), key) == plaintext


def test_encrypt_empty_string_round_trip():
    key = generate_key()
    assert decrypt_field(encrypt_field("", key), key) == ""


def test_encrypt_produces_different_ciphertext_each_call():
    """Nonce is random; same plaintext + key must yield different ciphertext."""
    key = generate_key()
    a = encrypt_field("hello", key)
    b = encrypt_field("hello", key)
    assert a != b
    assert decrypt_field(a, key) == "hello"
    assert decrypt_field(b, key) == "hello"


def test_decrypt_wrong_key_raises():
    k1 = generate_key()
    k2 = generate_key()
    ct = encrypt_field("secret", k1)
    with pytest.raises(DecryptionError):
        decrypt_field(ct, k2)


def test_decrypt_tampered_ciphertext_raises():
    key = generate_key()
    ct = encrypt_field("secret", key)
    # Flip last character before tag — flips ciphertext byte
    tampered = ct[:-2] + ("A" if ct[-2] != "A" else "B") + ct[-1]
    with pytest.raises(DecryptionError):
        decrypt_field(tampered, key)


def test_decrypt_unknown_scheme_raises():
    key = generate_key()
    with pytest.raises(DecryptionError):
        decrypt_field("v9:bogus", key)


def test_generate_key_length_32_bytes():
    key = generate_key()
    assert len(key) == 32


# --- Task 2.2: Keyring-backed key resolution ---

from anon_proxy.crypto import (
    KEYRING_SERVICE,
    KEYRING_USERNAME,
    KEY_ENV_VAR,
    KeyNotFoundError,
    resolve_key,
    store_key,
)


@pytest.fixture
def fake_keyring(monkeypatch):
    """Replace `keyring` with a thread-local in-memory dict."""
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


def test_store_and_resolve_key_via_keyring(fake_keyring, monkeypatch):
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    key = generate_key()
    store_key(key)
    assert resolve_key() == key


def test_resolve_key_falls_back_to_env_var(fake_keyring, monkeypatch):
    """When keyring has no entry, fall back to ANON_PROXY_TELEMETRY_KEY."""
    key = generate_key()
    monkeypatch.setenv(KEY_ENV_VAR, base64.urlsafe_b64encode(key).decode("ascii"))
    assert resolve_key() == key


def test_resolve_key_raises_when_neither_present(fake_keyring, monkeypatch):
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    with pytest.raises(KeyNotFoundError):
        resolve_key()


def test_resolve_key_keyring_takes_precedence_over_env(fake_keyring, monkeypatch):
    keyring_key = generate_key()
    env_key = generate_key()
    store_key(keyring_key)
    monkeypatch.setenv(KEY_ENV_VAR, base64.urlsafe_b64encode(env_key).decode("ascii"))
    assert resolve_key() == keyring_key


def test_keyring_constants_match_documented_naming():
    assert KEYRING_SERVICE == "anon-proxy"
    assert KEYRING_USERNAME == "telemetry"
    assert KEY_ENV_VAR == "ANON_PROXY_TELEMETRY_KEY"
