"""Field-level AES-256-GCM encryption for telemetry text fields.

Wire format: `v1:<base64url(nonce || ciphertext || tag)>` where nonce is
12 bytes (AESGCM standard) and tag is the 16-byte GCM tag appended by the
library. The `v1:` prefix lets future schemes coexist for migration.

Operability principle: only text-bearing fields are encrypted. Labels,
lengths, scores, and timestamps stay cleartext so `tail -f`, `wc -l`, and
`jq` keep working without keychain access.
"""

from __future__ import annotations

import base64
import os
import secrets

import keyring  # cross-platform; macOS Keychain, Linux Secret Service, Windows Credential Manager

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


SCHEME_PREFIX = "v1:"
NONCE_BYTES = 12
KEY_BYTES = 32  # AES-256

KEYRING_SERVICE = "anon-proxy"
KEYRING_USERNAME = "telemetry"
KEY_ENV_VAR = "ANON_PROXY_TELEMETRY_KEY"


class DecryptionError(Exception):
    """Raised when ciphertext cannot be decrypted (wrong key, tamper, bad scheme)."""


class KeyNotFoundError(Exception):
    """Raised when neither keyring nor env var has a usable key."""


def generate_key() -> bytes:
    """Generate a fresh 256-bit AES key."""
    return secrets.token_bytes(KEY_BYTES)


def encrypt_field(plaintext: str, key: bytes) -> str:
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes, got {len(key)}")
    nonce = secrets.token_bytes(NONCE_BYTES)
    aes = AESGCM(key)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    blob = nonce + ct
    return SCHEME_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii")


def decrypt_field(token: str, key: bytes) -> str:
    if not token.startswith(SCHEME_PREFIX):
        raise DecryptionError(f"unknown scheme prefix: {token[:8]!r}")
    if len(key) != KEY_BYTES:
        raise DecryptionError(f"key must be {KEY_BYTES} bytes, got {len(key)}")
    try:
        blob = base64.urlsafe_b64decode(token[len(SCHEME_PREFIX):].encode("ascii"))
    except Exception as e:
        raise DecryptionError(f"base64 decode failed: {e}") from e
    if len(blob) < NONCE_BYTES + 16:
        raise DecryptionError("ciphertext too short")
    nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    try:
        pt = AESGCM(key).decrypt(nonce, ct, None)
    except InvalidTag as e:
        raise DecryptionError("AESGCM tag verification failed (wrong key or tampered)") from e
    return pt.decode("utf-8")


def store_key(key: bytes) -> None:
    """Store the AES key in the OS keyring as base64url(key)."""
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes")
    keyring.set_password(
        KEYRING_SERVICE,
        KEYRING_USERNAME,
        base64.urlsafe_b64encode(key).decode("ascii"),
    )


def resolve_key() -> bytes:
    """Return the AES key, preferring keyring over env var. Raise if neither exists."""
    encoded = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if encoded is None:
        encoded = os.environ.get(KEY_ENV_VAR)
    if not encoded:
        raise KeyNotFoundError(
            f"No telemetry encryption key found. Set in keyring (service={KEYRING_SERVICE!r}, "
            f"user={KEYRING_USERNAME!r}) or env var {KEY_ENV_VAR}."
        )
    try:
        key = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception as e:
        raise KeyNotFoundError(f"key decode failed: {e}") from e
    if len(key) != KEY_BYTES:
        raise KeyNotFoundError(f"key must be {KEY_BYTES} bytes, got {len(key)}")
    return key


def ensure_key_exists() -> bytes:
    """Resolve, or generate-and-store on first run. Returns the key."""
    try:
        return resolve_key()
    except KeyNotFoundError:
        key = generate_key()
        store_key(key)
        return key
