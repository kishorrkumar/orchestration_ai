"""
Binary WebSocket Protocol framing for NVIDIA PersonaPlex / Moshi architecture.

Byte 0:
  0x00: Handshake
  0x01: Audio chunk (Opus packet or raw PCM bytes)
  0x02: Text chunk (UTF-8)
  0x03: Control action
  0x04: Metadata (JSON)
  0x05: Error (UTF-8)
  0x06: Ping
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Union


class MessageType(IntEnum):
    HANDSHAKE = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    CONTROL = 0x03
    METADATA = 0x04
    ERROR = 0x05
    PING = 0x06


class ControlAction(IntEnum):
    START = 0x00
    END_TURN = 0x01
    PAUSE = 0x02
    RESTART = 0x03


@dataclass(frozen=True)
class HandshakeMessage:
    version: int = 0
    model: int = 0

    @property
    def type(self) -> MessageType:
        return MessageType.HANDSHAKE


@dataclass(frozen=True)
class AudioMessage:
    data: bytes

    @property
    def type(self) -> MessageType:
        return MessageType.AUDIO


@dataclass(frozen=True)
class TextMessage:
    text: str

    @property
    def type(self) -> MessageType:
        return MessageType.TEXT


@dataclass(frozen=True)
class ControlMessage:
    action: ControlAction

    @property
    def type(self) -> MessageType:
        return MessageType.CONTROL


@dataclass(frozen=True)
class MetadataMessage:
    data: dict[str, Any]

    @property
    def type(self) -> MessageType:
        return MessageType.METADATA


@dataclass(frozen=True)
class ErrorMessage:
    error: str

    @property
    def type(self) -> MessageType:
        return MessageType.ERROR


@dataclass(frozen=True)
class PingMessage:
    @property
    def type(self) -> MessageType:
        return MessageType.PING


WSMessage = Union[
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ControlMessage,
    MetadataMessage,
    ErrorMessage,
    PingMessage,
]


def encode_message(message: WSMessage) -> bytes:
    """Encode a WSMessage into binary wire format."""
    if isinstance(message, HandshakeMessage):
        return bytes([MessageType.HANDSHAKE.value, message.version & 0xFF, message.model & 0xFF])
    elif isinstance(message, AudioMessage):
        return bytes([MessageType.AUDIO.value]) + message.data
    elif isinstance(message, TextMessage):
        return bytes([MessageType.TEXT.value]) + message.text.encode("utf-8")
    elif isinstance(message, ControlMessage):
        return bytes([MessageType.CONTROL.value, message.action.value & 0xFF])
    elif isinstance(message, MetadataMessage):
        return bytes([MessageType.METADATA.value]) + json.dumps(message.data).encode("utf-8")
    elif isinstance(message, ErrorMessage):
        return bytes([MessageType.ERROR.value]) + message.error.encode("utf-8")
    elif isinstance(message, PingMessage):
        return bytes([MessageType.PING.value])
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


MAX_MESSAGE_SIZE: int = 4 * 1024 * 1024  # 4 MB max payload


def decode_message(payload: bytes) -> WSMessage:
    """Decode a binary payload into the corresponding WSMessage."""
    if not payload:
        raise ValueError("Cannot decode empty payload")
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Payload exceeds maximum allowed size ({len(payload)} > {MAX_MESSAGE_SIZE})")

    kind = payload[0]
    body = payload[1:]

    if kind == MessageType.HANDSHAKE.value:
        # Handshake might have 0 bytes (server ack), 1 byte, or 2 bytes
        version = body[0] if len(body) > 0 else 0
        model = body[1] if len(body) > 1 else 0
        return HandshakeMessage(version=version, model=model)
    elif kind == MessageType.AUDIO.value:
        return AudioMessage(data=body)
    elif kind == MessageType.TEXT.value:
        text = body.decode("utf-8", errors="replace")
        return TextMessage(text=text)
    elif kind == MessageType.CONTROL.value:
        if not body:
            raise ValueError("Control message missing action byte")
        try:
            action = ControlAction(body[0])
        except ValueError:
            action = ControlAction.START
        return ControlMessage(action=action)
    elif kind == MessageType.METADATA.value:
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except Exception as e:
            data = {"raw": body.decode("utf-8", errors="replace"), "parse_error": str(e)}
        return MetadataMessage(data=data)
    elif kind == MessageType.ERROR.value:
        return ErrorMessage(error=body.decode("utf-8", errors="replace"))
    elif kind == MessageType.PING.value:
        return PingMessage()
    else:
        raise ValueError(f"Unknown message kind byte: {hex(kind)}")
