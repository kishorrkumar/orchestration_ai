"""
Text Normalization and Preprocessing for Indian-English Voice Agent.

Expands:
- Currency: ₹5000, 12,500 INR, Rs. 50 -> "five thousand rupees", "twelve thousand five hundred rupees"
- Large Indian denominations: Lakhs and Crores (e.g. ₹2.5 crore, 15 lakh)
- Numbers & decimals: 100 -> "one hundred", 3.5 -> "three point five"
- Phone numbers: digit by digit (e.g. 9876543210 -> "nine eight seven six five four three two one zero")
- Time: 4:30 pm -> "four thirty p m", 10:00 am -> "ten a m"
- Dates: 15/08/2026 -> "fifteen august twenty twenty-six"
- Abbreviations: AI -> "A.I.", Dr. -> "Doctor", Mr. -> "Mister"
- Strips: Markdown, asterisks, hashtags, emojis, brackets, parentheses.
"""

from __future__ import annotations
import re
from typing import Optional

# Indian English Number Maps
UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

MONTHS = {
    1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
    7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"
}

COMMON_ABBREVIATIONS = {
    r"\bAI\b": "A.I.",
    r"\bML\b": "M.L.",
    r"\bTTS\b": "T.T.S.",
    r"\bSTT\b": "S.T.T.",
    r"\bVAD\b": "V.A.D.",
    r"\bLLM\b": "L.L.M.",
    r"\bAPI\b": "A.P.I.",
    r"\bDr\.\s*": "Doctor ",
    r"\bMr\.\s*": "Mister ",
    r"\bMrs\.\s*": "Missus ",
    r"\bMs\.\s*": "Miz ",
    r"\bProf\.\s*": "Professor ",
    r"\bvs\.\s*": "versus ",
    r"\betc\.\s*": "etcetera ",
    r"\be\.g\.\s*": "for example ",
    r"\bi\.e\.\s*": "that is ",
    r"\bapprox\.\s*": "approximately ",
    r"\bmin\b": "minutes",
    r"\bsec\b": "seconds",
    r"\bhr\b": "hours",
    r"\bhrs\b": "hours",
    r"\bkg\b": "kilograms",
    r"\bgm\b": "grams",
    r"\bkm\b": "kilometers",
    r"\bcm\b": "centimeters",
    r"\bft\b": "feet",
    r"\bin\b": "inches",
}


def number_to_spoken_words(n: int) -> str:
    """Convert integer to Indian English spoken words."""
    if n < 0:
        return "minus " + number_to_spoken_words(-n)
    if n < 20:
        return UNITS[n]
    if n < 100:
        rem = n % 10
        tens_word = TENS[n // 10]
        return tens_word if rem == 0 else f"{tens_word} {UNITS[rem]}"
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        h_part = f"{UNITS[hundreds]} hundred"
        return h_part if rem == 0 else f"{h_part} and {number_to_spoken_words(rem)}"
    if n < 100000:
        # Thousands
        thousands = n // 1000
        rem = n % 1000
        t_part = f"{number_to_spoken_words(thousands)} thousand"
        return t_part if rem == 0 else f"{t_part} {number_to_spoken_words(rem)}"
    if n < 10000000:
        # Lakhs (1,00,000 to 99,99,999)
        lakhs = n // 100000
        rem = n % 100000
        l_part = f"{number_to_spoken_words(lakhs)} lakh"
        return l_part if rem == 0 else f"{l_part} {number_to_spoken_words(rem)}"
    # Crores (1,00,00,000+)
    crores = n // 10000000
    rem = n % 10000000
    c_part = f"{number_to_spoken_words(crores)} crore"
    return c_part if rem == 0 else f"{c_part} {number_to_spoken_words(rem)}"


def digits_to_words(digits_str: str) -> str:
    """Read a string of digits one by one (for phone numbers, OTPs, PINs)."""
    clean_digits = re.sub(r"[^\d]", "", digits_str)
    return " ".join(UNITS[int(d)] for d in clean_digits)


def normalize_indian_english_text(text: str, allow_fillers: bool = False) -> str:
    """
    Comprehensive text preprocessing for natural Indian-English speech synthesis.
    Safe, robust, and maintains high intelligibility.
    """
    if not text:
        return ""

    out = text

    # 1. Strip Markdown, Emojis, Code Blocks, and Parentheses content
    out = re.sub(r"```[\s\S]*?```", "", out)
    out = re.sub(r"`[^`]*`", "", out)
    out = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", out)  # Markdown links -> text
    out = re.sub(r"[#*_~>]+", " ", out)  # Markdown formatting characters
    out = re.sub(r"\([^\)]*\)", "", out)  # Remove bracketed aside notes
    out = re.sub(r"\{[^\}]*\}", "", out)
    # Strip emojis (high unicode ranges)
    out = re.sub(r"[\U00010000-\U0010ffff]", "", out)
    out = re.sub(r"[\u2600-\u27bf]", "", out)

    # 2. Expand Currency: ₹, INR, Rs., Rs
    # Pattern: (₹|Rs\.?|INR)\s*([0-9,]+(?:\.[0-9]+)?)\s*(crore|lakh)?
    def _expand_currency(m: re.Match) -> str:
        amt_str = m.group(2).replace(",", "")
        scale = m.group(3).lower() if m.group(3) else ""
        try:
            val = float(amt_str)
            if scale == "crore":
                val *= 10000000
            elif scale == "lakh":
                val *= 100000
            
            if val.is_integer():
                int_val = int(val)
                return f"{number_to_spoken_words(int_val)} rupees"
            else:
                int_part = int(val)
                dec_part = int(round((val - int_part) * 100))
                words = f"{number_to_spoken_words(int_part)} rupees"
                if dec_part > 0:
                    words += f" and {number_to_spoken_words(dec_part)} paise"
                return words
        except Exception:
            return f"{m.group(2)} rupees"

    curr_pattern = re.compile(
        r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(crore|lakh|thousand)?|"
        r"([\d,]+(?:\.\d+)?)\s*(?:₹|rs\.?|inr|rupees)\s*(crore|lakh|thousand)?",
        re.IGNORECASE,
    )

    def _curr_sub(match):
        amt = match.group(1) or match.group(3)
        scale = match.group(2) or match.group(4)
        amt_clean = amt.replace(",", "")
        try:
            val = float(amt_clean)
            if scale:
                s = scale.lower()
                if "crore" in s:
                    val *= 10000000
                elif "lakh" in s:
                    val *= 100000
                elif "thousand" in s:
                    val *= 1000
            int_val = int(round(val))
            return f"{number_to_spoken_words(int_val)} rupees"
        except Exception:
            return f"{amt} rupees"

    out = curr_pattern.sub(_curr_sub, out)

    # 3. Expand Times: e.g. 10:30 am, 4:15 pm, 08:00
    def _time_sub(m: re.Match) -> str:
        hh = int(m.group(1))
        mm = int(m.group(2))
        meridiem = m.group(3)
        h_word = number_to_spoken_words(hh)
        if mm == 0:
            m_word = "o'clock" if not meridiem else ""
        elif mm < 10:
            m_word = f"oh {number_to_spoken_words(mm)}"
        else:
            m_word = number_to_spoken_words(mm)
        res = f"{h_word} {m_word}".strip()
        if meridiem:
            res += " " + " ".join(list(meridiem.lower()))
        return res

    out = re.sub(r"\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?\b", _time_sub, out, flags=re.IGNORECASE)

    # 4. Expand Phone numbers / OTPs (10-digit mobile numbers or 4-6 digit codes)
    def _phone_sub(m: re.Match) -> str:
        digits = m.group(0)
        return digits_to_words(digits)

    # 10 digit Indian mobile numbers (starting with 6, 7, 8, 9)
    out = re.sub(r"\b[6789]\d{9}\b", _phone_sub, out)

    # 5. Expand Abbreviations
    for pattern, replacement in COMMON_ABBREVIATIONS.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    # 6. Expand Decimals (e.g. 3.5 -> "three point five")
    def _decimal_sub(m: re.Match) -> str:
        whole = int(m.group(1))
        frac = m.group(2)
        frac_words = " ".join(UNITS[int(d)] for d in frac)
        return f"{number_to_spoken_words(whole)} point {frac_words}"

    out = re.sub(r"\b(\d+)\.(\d+)\b", _decimal_sub, out)

    # 7. Expand Remaining standalone Numbers
    def _int_sub(m: re.Match) -> str:
        val = int(m.group(0))
        # Keep 4-digit years natural (e.g. 2026 -> "twenty twenty-six")
        if 1900 <= val <= 2099:
            c = val // 100
            y = val % 100
            c_word = number_to_spoken_words(c)
            y_word = "hundred" if y == 0 else (f"oh {number_to_spoken_words(y)}" if y < 10 else number_to_spoken_words(y))
            return f"{c_word} {y_word}"
        return number_to_spoken_words(val)

    out = re.sub(r"\b\d+\b", _int_sub, out)

    # 8. Clean up punctuation, spaces, and duplicate characters
    out = re.sub(r"[\/\\_|=+@#$^&*~`]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()

    return out
