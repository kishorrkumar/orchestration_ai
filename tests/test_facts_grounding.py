"""
Tests for Snapserve sales facts grounding, pricing defense, and monologue limits.
"""

import json
from pathlib import Path
import pytest
import re

from orchestration.prompts.compiler import compile_prompt, lint_prompt
from orchestration.audio.detokenizer import detokenize_sentencepiece_stream


def test_facts_sheet_exists_and_valid():
    facts_file = Path("orchestration/prompts/facts_sheet.json")
    assert facts_file.exists(), "facts_sheet.json must exist"
    with open(facts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["company_name"] == "Snapserve"
    assert "pricing_policy" in data
    assert "NEVER" in data["pricing_policy"]["quote_policy"]


def test_anti_hallucination_pricing_response():
    """Verify that pricing defense redirects to demo without inventing price numbers."""
    test_pricing_turn = "It depends on your monthly call volume, but I can set up a quick 10-minute demo to walk through exact pricing."
    assert not re.search(r"\$\d+|\b\d+\s*dollars|\bhundred dollars\b", test_pricing_turn, flags=re.IGNORECASE)
    assert "demo" in test_pricing_turn.lower()


def test_monologue_cutoff_logic():
    """Verify monologue guard flag logic on verbose agent speech."""
    short_speech = "Got it, that makes sense. How many inbound calls do you get a week?"
    short_words = short_speech.split()
    assert len(short_words) <= 25

    verbose_speech = "Well let me tell you everything about our product we have hundreds of features and we also connect with all your databases and we can do outbound calls and inbound calls and schedule appointments and we can even do billing reminders for your customers."
    verbose_words = verbose_speech.split()
    assert len(verbose_words) > 25, "Verbose speech should exceed 25-word conversational turn threshold"


def test_anti_moshi_leak_in_runtime_stream():
    """Verify that detokenizer eradicates foundation Moshi leaks."""
    leaked_tokens = [" Hello", ",", " My", " name", " is", " Moshi", "."]
    cleaned = detokenize_sentencepiece_stream(leaked_tokens, agent_name="Ananya")
    assert "Moshi" not in cleaned
    assert "Ananya" in cleaned
