"""
Text Normalization for Spoken Dialogue and TTS Synthesis.

Prepares raw LLM output for natural acoustic prosody:
- Strips markdown formatting (*, _, #, `, [], ())
- Strips emojis and pictographs
- Expands Indian currency (Rs., ₹, INR) and numbers (lakh, crore)
- Expands tech and Indian acronyms (UPI -> U P I, AI -> A I, IPL -> I P L, etc.)
- Converts ellipsis (...) into natural prosodic pauses
- Cleans whitespace and maintains natural comma cadences
"""

from __future__ import annotations

import re

# Indian English numbers map
NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety", 100: "one hundred",
}

# Currency regex
CURRENCY_INR_REGEX = re.compile(
    r"(?:(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(crore|lakh|thousand)?|"
    r"([\d,]+(?:\.\d+)?)\s*(crore|lakh|thousand)?\s*(?:₹|rs\.?|inr|rupees))",
    re.IGNORECASE,
)

# Common spoken abbreviations to space out for clear TTS letter pronunciation
ABBREVIATIONS = {
    r"\bUPI\b": "U P I",
    r"\bAI\b": "A.I.",
    r"\bML\b": "M.L.",
    r"\bIPL\b": "I P L",
    r"\bAC\b": "A C",
    r"\bATM\b": "A T M",
    r"\bOTP\b": "O T P",
    r"\bQR\b": "Q R",
    r"\bID\b": "I D",
    r"\bSMS\b": "S M S",
    r"\bKYC\b": "K Y C",
    r"\bGST\b": "G S T",
    r"\bRBI\b": "R B I",
    r"\bSIM\b": "sim",
}


def number_to_words(n: int) -> str:
    """Convert integer to spoken words for common conversational numbers."""
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return f"{NUM_WORDS.get(tens, '')} {NUM_WORDS.get(ones, '')}".strip()
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        if rem == 0:
            return f"{NUM_WORDS.get(hundreds, str(hundreds))} hundred"
        return f"{NUM_WORDS.get(hundreds, str(hundreds))} hundred {number_to_words(rem)}"
    return str(n)


def normalize_currency_inr(text: str) -> str:
    """Normalize Indian Rupee notations into clean spoken words."""
    def _inr_sub(match):
        amount = match.group(1) or match.group(3)
        scale = (match.group(2) or match.group(4) or "").lower()
        amount_clean = amount.replace(",", "")
        try:
            num = float(amount_clean)
            if scale == "crore":
                num *= 10000000
            elif scale == "lakh":
                num *= 100000
            elif scale == "thousand":
                num *= 1000

            num = int(round(num))
            if num >= 10000000:
                crores = int(num / 10000000)
                rem = num % 10000000
                if rem == 0:
                    return f" {number_to_words(crores)} crore rupees "
                return f" {number_to_words(crores)} crore {number_to_words(rem)} rupees "
            elif num >= 100000:
                lakhs = int(num / 100000)
                rem = num % 100000
                if rem == 0:
                    return f" {number_to_words(lakhs)} lakh rupees "
                if rem == 50000:
                    return f" {number_to_words(lakhs)} point five lakh rupees "
                return f" {number_to_words(lakhs)} lakh {number_to_words(rem)} rupees "
            elif num >= 1000:
                thousands = int(num / 1000)
                rem = num % 1000
                if rem == 0:
                    return f" {number_to_words(thousands)} thousand rupees "
                return f" {number_to_words(thousands)} thousand {number_to_words(rem)} rupees "
            else:
                return f" {number_to_words(num)} rupees "
        except ValueError:
            return f" {amount_clean} rupees "

    return CURRENCY_INR_REGEX.sub(_inr_sub, text)


def strip_markdown_and_emojis(text: str) -> str:
    """Strip all markdown formatting, brackets, stage directions, and emojis."""
    if not text:
        return ""

    s = text

    # Remove code blocks and inline code
    s = re.sub(r"```[\s\S]*?```", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Remove bold, italics, strikethrough: **text**, *text*, __text__, _text_, ~~text~~
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"~~([^~]+)~~", r"\1", s)

    # Remove headers: # Header
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)

    # Remove markdown link brackets [Title](url) -> Title
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Remove bullet points / list numbers at start of lines
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)

    # Remove bracketed stage directions e.g. [laughs], (chuckles)
    s = re.sub(r"\[[\w\s]+\]", "", s)
    s = re.sub(r"\([\w\s]+ing\)", "", s)

    # Remove emojis and non-ascii pictographs
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    s = re.sub(r"[\u2600-\u27bf]", "", s)
    s = re.sub(r"[\u200d\ufe0f]", "", s)

    return s


def normalize_for_tts(text: str) -> str:
    """Complete text normalization pipeline for natural spoken audio."""
    if not text:
        return ""

    # 1. Clean markdown & emojis
    s = strip_markdown_and_emojis(text)

    # 2. Expand Indian currency
    s = normalize_currency_inr(s)

    # 3. Expand common technical and Indian abbreviations
    for pattern, replacement in ABBREVIATIONS.items():
        s = re.sub(pattern, replacement, s)

    # 4. Turn ellipsis into prosodic commas
    s = re.sub(r"\.{2,}|…", ", ", s)

    # 5. Clean up awkward spacing before punctuation
    s = re.sub(r"\s+([,;.?!])", r"\1", s)

    # 6. Compress multiple whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 7. Conversational voice constraint: keep to concise spoken length (<= 50 words)
    words = s.split()
    if len(words) > 48:
        sentences = re.split(r"(?<=[.!?])\s+", s)
        acc = []
        count = 0
        for sent in sentences:
            sent_words = len(sent.split())
            if count + sent_words <= 48:
                acc.append(sent)
                count += sent_words
            else:
                break
        if acc:
            s = " ".join(acc).strip()
        else:
            s = " ".join(words[:45]).strip() + "."

    return s
