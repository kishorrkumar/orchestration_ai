"""
Unit tests for Spoken Text Normalization for TTS and Indian English Prosody.
"""

from orchestration.chunker.normalizer import (
    normalize_currency_inr,
    normalize_for_tts,
    strip_markdown_and_emojis,
)


def test_strip_markdown_and_emojis():
    raw = "**Hello**! This is _Aarav_ with `code` and [link](https://example.com) 😊🚀."
    cleaned = strip_markdown_and_emojis(raw)
    assert "**" not in cleaned
    assert "_" not in cleaned
    assert "`" not in cleaned
    assert "https://" not in cleaned
    assert "😊" not in cleaned
    assert "🚀" not in cleaned
    assert "Hello! This is Aarav with code and link" in cleaned


def test_normalize_currency_inr():
    assert "five hundred rupees" in normalize_currency_inr("The ticket is ₹500.").lower()
    assert "two thousand rupees" in normalize_currency_inr("Pay Rs. 2000 now.").lower()
    assert "ten lakh rupees" in normalize_currency_inr("It costs ₹10 lakh.").lower()
    assert "one crore rupees" in normalize_currency_inr("The budget is 1 crore rupees.").lower()


def test_normalize_for_tts_pipeline():
    raw = "Sure! I can help with UPI and A.I. payments... Just send ₹1500 to my ID 👍."
    norm = normalize_for_tts(raw)
    assert "U P I" in norm
    assert "A.I." in norm
    assert "fifteen hundred rupees" in norm or "one thousand five hundred rupees" in norm
    assert "👍" not in norm
    assert "..." not in norm
    assert "," in norm  # Ellipsis converted to pause comma
