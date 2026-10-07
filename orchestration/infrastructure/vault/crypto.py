"""
Cryptographic Vault for Engine B Provider Credentials.
Uses AES-256-GCM with HKDF-derived keys from APP_SECRET_KEY.
Ensures zero plaintext storage, key rotation, and strict tamper resistance.
"""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path
from typing import Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VAULT_SALT = b"persona_engine_b_vault_salt_v1"
VAULT_INFO = b"persona_provider_credentials_aes_256_gcm"
DEFAULT_KEY_VERSION = 1
DEV_SECRETS_DIR = Path(".secrets")
DEV_SECRET_FILE = DEV_SECRETS_DIR / "app_secret.key"


def get_app_secret(key_version: int = DEFAULT_KEY_VERSION) -> bytes:
    """
    Retrieves or derives the 32-byte secret for a specific key_version.
    In production (ENV=production), APP_SECRET_KEY must be provided via environment.
    In development, auto-generates a key into ignored .secrets/app_secret.key.
    """
    env_name = os.environ.get("ENV", "development").lower()

    # Check for version-specific key e.g. APP_SECRET_KEY_v1 or generic APP_SECRET_KEY
    env_key = os.environ.get(f"APP_SECRET_KEY_V{key_version}") or os.environ.get("APP_SECRET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    if env_name == "production":
        raise RuntimeError(
            f"APP_SECRET_KEY (or APP_SECRET_KEY_V{key_version}) is required in production environment"
        )

    # Local development auto-generation
    DEV_SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    if not DEV_SECRET_FILE.exists():
        new_secret = secrets.token_hex(32)
        DEV_SECRET_FILE.write_text(new_secret, encoding="utf-8")
        print(f"[SECURITY NOTICE] Auto-generated dev APP_SECRET_KEY in {DEV_SECRET_FILE.as_posix()}")
        return new_secret.encode("utf-8")

    return DEV_SECRET_FILE.read_text(encoding="utf-8").strip().encode("utf-8")


def derive_aes_key(secret: bytes, salt: bytes = VAULT_SALT, info: bytes = VAULT_INFO) -> bytes:
    """Derives a 32-byte (256-bit) AES key using HKDF-SHA256."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=info,
    )
    return hkdf.derive(secret)


def extract_last4(plaintext: str) -> str:
    """Extracts last 4 characters of the key for safe UI identification."""
    clean = plaintext.strip()
    return clean[-4:] if len(clean) >= 4 else clean


def encrypt_api_key(
    plaintext_key: str,
    key_version: int = DEFAULT_KEY_VERSION,
    secret: bytes | None = None,
) -> Tuple[str, str, str, int]:
    """
    Encrypts an API key using AES-256-GCM.
    Returns: (ciphertext_b64, nonce_b64, last4, key_version)
    """
    if not plaintext_key:
        raise ValueError("Cannot encrypt an empty key")

    raw_secret = secret if secret is not None else get_app_secret(key_version)
    aes_key = derive_aes_key(raw_secret)
    aesgcm = AESGCM(aes_key)

    nonce = secrets.token_bytes(12)  # Standard 96-bit nonce for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext_key.encode("utf-8"), associated_data=None)

    last4 = extract_last4(plaintext_key)
    ciphertext_b64 = base64.b64encode(ciphertext).decode("utf-8")
    nonce_b64 = base64.b64encode(nonce).decode("utf-8")

    return ciphertext_b64, nonce_b64, last4, key_version


def decrypt_api_key(
    ciphertext_b64: str,
    nonce_b64: str,
    key_version: int = DEFAULT_KEY_VERSION,
    secret: bytes | None = None,
) -> str:
    """
    Decrypts an AES-256-GCM encrypted API key.
    Raises cryptography.exceptions.InvalidTag on tamper or wrong key.
    """
    raw_secret = secret if secret is not None else get_app_secret(key_version)
    aes_key = derive_aes_key(raw_secret)
    aesgcm = AESGCM(aes_key)

    ciphertext = base64.b64decode(ciphertext_b64.encode("utf-8"))
    nonce = base64.b64decode(nonce_b64.encode("utf-8"))

    plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext_bytes.decode("utf-8")
