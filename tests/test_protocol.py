import pytest
from orchestration.protocol.messages import (
    MessageType,
    ControlAction,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ControlMessage,
    MetadataMessage,
    ErrorMessage,
    PingMessage,
    encode_message,
    decode_message,
)


def test_handshake_roundtrip():
    msg = HandshakeMessage(version=0, model=0)
    encoded = encode_message(msg)
    assert encoded == bytes([0x00, 0x00, 0x00])
    decoded = decode_message(encoded)
    assert isinstance(decoded, HandshakeMessage)
    assert decoded.version == 0
    assert decoded.model == 0


def test_handshake_server_ack():
    # PersonaPlex server sends a single 0x00 byte as handshake ack
    decoded = decode_message(bytes([0x00]))
    assert isinstance(decoded, HandshakeMessage)
    assert decoded.version == 0
    assert decoded.model == 0


def test_audio_roundtrip():
    dummy_audio = b"\x01\x02\x03\x04\x05"
    msg = AudioMessage(data=dummy_audio)
    encoded = encode_message(msg)
    assert encoded[0] == 0x01
    assert encoded[1:] == dummy_audio

    decoded = decode_message(encoded)
    assert isinstance(decoded, AudioMessage)
    assert decoded.data == dummy_audio


def test_text_roundtrip():
    text = "Hello, world! I am PersonaPlex."
    msg = TextMessage(text=text)
    encoded = encode_message(msg)
    assert encoded[0] == 0x02

    decoded = decode_message(encoded)
    assert isinstance(decoded, TextMessage)
    assert decoded.text == text


def test_control_roundtrip():
    for action in ControlAction:
        msg = ControlMessage(action=action)
        encoded = encode_message(msg)
        assert encoded[0] == 0x03
        assert encoded[1] == action.value

        decoded = decode_message(encoded)
        assert isinstance(decoded, ControlMessage)
        assert decoded.action == action


def test_metadata_roundtrip():
    meta = {"session_id": "test-123", "sample_rate": 24000, "barge_in": True}
    msg = MetadataMessage(data=meta)
    encoded = encode_message(msg)
    assert encoded[0] == 0x04

    decoded = decode_message(encoded)
    assert isinstance(decoded, MetadataMessage)
    assert decoded.data == meta


def test_error_roundtrip():
    err_text = "Capacity exceeded: no workers available"
    msg = ErrorMessage(error=err_text)
    encoded = encode_message(msg)
    assert encoded[0] == 0x05

    decoded = decode_message(encoded)
    assert isinstance(decoded, ErrorMessage)
    assert decoded.error == err_text


def test_ping_roundtrip():
    msg = PingMessage()
    encoded = encode_message(msg)
    assert encoded == bytes([0x06])

    decoded = decode_message(encoded)
    assert isinstance(decoded, PingMessage)


def test_empty_payload_fails():
    with pytest.raises(ValueError, match="Cannot decode empty payload"):
        decode_message(b"")


def test_invalid_kind_fails():
    with pytest.raises(ValueError, match="Unknown message kind"):
        decode_message(b"\xFF\x01\x02")
