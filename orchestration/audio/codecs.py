"""
Telephony Codecs for Voice Agent Platform.
Provides bit-exact ITU-T G.711 mu-law encoder and decoder with O(1) table lookups.
"""

from __future__ import annotations

import base64
import zlib
from typing import Union

import numpy as np


# ITU-T G.711 mu-law decode lookup table (256 entries mapping 8-bit mu-law -> int16 PCM)
def _generate_decode_table() -> np.ndarray:
    table = np.empty(256, dtype=np.int16)
    for u_byte in range(256):
        u = (~u_byte) & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        linear = ((mantissa << 3) + 0x84) << exponent
        linear -= 0x84
        table[u_byte] = -linear if (sign != 0) else linear
    return table

_ULAW_DECODE_TABLE = _generate_decode_table()

# Precomputed 65536-entry encode lookup table (mapping 16-bit PCM -> uint8 mu-law)
# Decompressed from compact zlib-compressed base64 string
_ULAW_ENCODE_B64 = (
    "eNrtwVf8EGIUx+F/iigSiYgWGrKVMkqSklIRQrIjLSlJUR+UMkplpGE0qYRKaAtFISurjdIwIpqUcfFenZv34lyci9/3eQoKACBWI"
    "QCy9gEgqzAAWUUAyNoXgKz9AMgqCkDW/gBkHQBAVjEAsooDkHUgAFkHAZBVAoCsgwHIKglph0DaoZBWCtIOg7TSkHY4pB0BaWUg7"
    "UhIOwrSykLa0ZB2DKSVg7TykFZBXEVxlcQdK+44cceLqyyuiriq4qqJO0FcdXEnijtJ3MniThF3arDTgp0e7IxgNYLVDHZmsFrBa"
    "gc7K9jZwc4Jdm6wOsHqBjsvWD2n853qO13g1MDpQqeGTo2cLnJq7HSxUxOnpk6XODVzau7UIuPSjMsyWmZcnnFFxpUZrTKuyrg6"
    "45qM1hnXZrTJuC7j+owbjBuNm4ybjVuMtsatxm1GO+N2o73RwehodDI6G3cYXZI7k65Jt+SupHtyd9IjuSfpmfRK7k3uS3onfZJ"
    "///dPsjfZk/yd/JXsTnYlO5MdyfZkW/Jn8keyNfk9+c3YYvxq/GL8bPxkbDY2GRuNDcaPxnpjnfGD8b3xnbE2Y03G6oxVGSszVm"
    "Qsz/g245uMrzO+yvgyY1nGFxmfZ3yW8anTJ05LnT52+sjpQ6clToudPnB632mR00Kn95zedXrHaYHT28HmB5sXbG6wOcFmB5sV"
    "bGawt4K9GeyNYDOCvR5serBpwaYGe03cq+JeETdF3MviJoubJG6iuJfEvShugrjx4saJGytujLjR4l6AtOch7TlIexbSRkHaSEgb"
    "AWnDIe0ZSBsGaU9D2lOQ9iSkPQFpQyFtCKQNBiDrcQCyBgGQNRCArMcAyHoUgKxHAMh6GICsAQBk9Qcg6yEAsvoBkNUXgKwHAch"
    "6AICs+wEg0H9nTueX"
)
_ULAW_ENCODE_TABLE = np.frombuffer(
    zlib.decompress(base64.b64decode(_ULAW_ENCODE_B64)),
    dtype=np.uint8,
)


def encode_ulaw(audio: Union[np.ndarray, bytes]) -> bytes:
    """
    Encode 16-bit PCM audio samples to ITU-T G.711 mu-law bytes.

    Args:
        audio: 1D numpy array of int16 or float32 (scaled [-1.0, 1.0]), or raw int16 PCM bytes.

    Returns:
        Raw mu-law encoded bytes.
    """
    if isinstance(audio, bytes):
        pcm16 = np.frombuffer(audio, dtype=np.int16)
    elif audio.dtype == np.float32 or audio.dtype == np.float64:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
    elif audio.dtype == np.int16:
        pcm16 = audio
    else:
        pcm16 = audio.astype(np.int16)

    indices = pcm16.astype(np.int32) + 32768
    return _ULAW_ENCODE_TABLE[indices].tobytes()


def decode_ulaw(ulaw_data: Union[bytes, np.ndarray]) -> np.ndarray:
    """
    Decode ITU-T G.711 mu-law bytes to 16-bit linear PCM numpy array.

    Args:
        ulaw_data: Bytes object or uint8 numpy array containing mu-law audio.

    Returns:
        1D numpy array of int16 PCM samples.
    """
    if isinstance(ulaw_data, bytes):
        raw = np.frombuffer(ulaw_data, dtype=np.uint8)
    elif ulaw_data.dtype != np.uint8:
        raw = ulaw_data.astype(np.uint8)
    else:
        raw = ulaw_data

    return _ULAW_DECODE_TABLE[raw]


def float32_to_ulaw(audio_f32: np.ndarray) -> bytes:
    """Converts float32 audio [-1.0, 1.0] to G.711 mu-law bytes."""
    return encode_ulaw(audio_f32)


def ulaw_to_float32(ulaw_data: Union[bytes, np.ndarray]) -> np.ndarray:
    """Converts G.711 mu-law bytes to float32 audio array [-1.0, 1.0]."""
    pcm16 = decode_ulaw(ulaw_data)
    return np.clip(pcm16.astype(np.float32) / 32767.0, -1.0, 1.0)
