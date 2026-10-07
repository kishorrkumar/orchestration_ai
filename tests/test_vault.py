"""
Tests for Cryptographic Vault and SSRF Protection.
"""

import pytest
from cryptography.exceptions import InvalidTag

from orchestration.infrastructure.vault.crypto import (
    decrypt_api_key,
    encrypt_api_key,
    extract_last4,
)
from orchestration.infrastructure.vault.ssrf import validate_base_url
from orchestration.shared.errors import ValidationError


def test_vault_roundtrip():
    secret = b"test_secret_key_32_bytes_long!!"
    key_text = "sk-ant-api03-abcdef1234567890"

    ciphertext_b64, nonce_b64, last4, ver = encrypt_api_key(key_text, key_version=1, secret=secret)
    assert last4 == "7890"
    assert ver == 1
    assert ciphertext_b64 != key_text
    assert key_text not in ciphertext_b64

    decrypted = decrypt_api_key(ciphertext_b64, nonce_b64, key_version=1, secret=secret)
    assert decrypted == key_text


def test_vault_tamper_detection():
    secret = b"test_secret_key_32_bytes_long!!"
    key_text = "dg_live_secret_key_123456"

    ciphertext_b64, nonce_b64, _, _ = encrypt_api_key(key_text, key_version=1, secret=secret)

    # Tamper with ciphertext by corrupting characters
    tampered_ct = ("A" if ciphertext_b64[0] != "A" else "B") + ciphertext_b64[1:]

    with pytest.raises(Exception):
        decrypt_api_key(tampered_ct, nonce_b64, key_version=1, secret=secret)


def test_vault_wrong_key():
    secret1 = b"test_secret_key_32_bytes_long_1"
    secret2 = b"test_secret_key_32_bytes_long_2"
    key_text = "cartesia_secret_key_9999"

    ciphertext_b64, nonce_b64, _, _ = encrypt_api_key(key_text, key_version=1, secret=secret1)

    with pytest.raises(InvalidTag):
        decrypt_api_key(ciphertext_b64, nonce_b64, key_version=1, secret=secret2)


def test_extract_last4():
    assert extract_last4("abcdef1234") == "1234"
    assert extract_last4("xyz") == "xyz"
    assert extract_last4("    1234    ") == "1234"


def test_vault_empty_key():
    with pytest.raises(ValueError):
        encrypt_api_key("", secret=b"secret_key_32_bytes_long!!")


def test_ssrf_validator_blocks_private_and_loopback():
    # Loopback
    with pytest.raises(ValidationError):
        validate_base_url("http://127.0.0.1:8000/v1", allow_local=False)

    with pytest.raises(ValidationError):
        validate_base_url("http://localhost:8000/v1", allow_local=False)

    # AWS metadata / link-local
    with pytest.raises(ValidationError):
        validate_base_url("http://169.254.169.254/latest/meta-data/", allow_local=False)

    # Private IP RFC 1918
    with pytest.raises(ValidationError):
        validate_base_url("http://192.168.1.50:11434/v1", allow_local=False)

    with pytest.raises(ValidationError):
        validate_base_url("http://10.0.0.5:8000/v1", allow_local=False)


def test_ssrf_validator_allows_public_urls():
    url = "https://api.openai.com/v1"
    assert validate_base_url(url, allow_local=False) == url

    deepgram_url = "https://api.deepgram.com"
    assert validate_base_url(deepgram_url, allow_local=False) == deepgram_url


def test_ssrf_validator_allows_local_when_flag_enabled():
    local_url = "http://127.0.0.1:11434/v1"
    assert validate_base_url(local_url, allow_local=True) == local_url
