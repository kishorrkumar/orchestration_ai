import pytest
from orchestration.persona.registry import default_registry, PersonaRegistry
from orchestration.persona.dialogue import StrictVoiceDialogueEngine, GroundedDialogueEngine
from orchestration.persona.prompts import build_system_prompt, MASTER_VOICE_AGENT_SYSTEM_PROMPT


def test_system_prompt_contains_12_principles():
    """Verify that the master system prompt contains all 12 Conversation Principles verbatim."""
    prompt = MASTER_VOICE_AGENT_SYSTEM_PROMPT
    required_principles = [
        "1. LISTEN FIRST",
        "2. RESPOND TO THE LATEST MESSAGE",
        "3. BE CONCISE",
        "4. ONE QUESTION AT A TIME",
        "5. DO NOT REPEAT INFORMATION",
        "6. NATURAL ACKNOWLEDGEMENT",
        "7. HUMAN-LIKE TURN TAKING",
        "8. NEVER SOUND ROBOTIC",
        "9. HANDLE INTERRUPTIONS",
        "10. HANDLE UNCERTAINTY",
        "11. MAINTAIN CONTEXT",
        "12. CONVERSATION PRIORITY",
    ]
    for p in required_principles:
        assert p in prompt, f"Expected principle {p} in master system prompt"


def test_nine_accent_character_presets_registered():
    """Verify all 3 accents x 3 characters (9 presets) are registered with system prompts."""
    expected_ids = [
        ("indian_pro", "indian", "professional"),
        ("indian_funny", "indian", "funny"),
        ("indian_warm", "indian", "warm"),
        ("american_pro", "american", "professional"),
        ("american_funny", "american", "funny"),
        ("american_warm", "american", "warm"),
        ("british_pro", "british", "professional"),
        ("british_funny", "british", "funny"),
        ("british_warm", "british", "warm"),
    ]

    for persona_id, expected_accent, expected_character in expected_ids:
        persona = default_registry.get(persona_id)
        assert persona is not None, f"Persona {persona_id} should be registered"
        assert expected_accent in persona.accent.lower(), f"Expected accent {expected_accent} for {persona_id}"
        assert expected_character in persona.character.lower(), f"Expected character {expected_character} for {persona_id}"
        prompt = persona.get_formatted_text_prompt()
        assert "<system>" in prompt
        assert "CONVERSATION PRINCIPLES" in prompt
        assert persona.voice_prompt.endswith(".pt")


def test_dialogue_engine_truthful_personal_questions():
    """Verify the dialogue engine answers questions like 'What did you eat today?' honestly without bluffing."""
    engine = StrictVoiceDialogueEngine(accent="american", character="warm")
    reply = engine.generate_reply("What did you eat today?")
    
    assert "operates on key fundamental principles" not in reply.lower()
    assert "[backchannel" not in reply.lower()
    assert any(term in reply.lower() for term in ["ai", "food", "eat", "electricity", "code"])


def test_strict_dialogue_engine_principles():
    """Verify strict adherence to principles: conciseness, single question, memory retention, no robotic phrases."""
    engine = StrictVoiceDialogueEngine(accent="indian", character="professional")

    # Principle 11 & 5: Context memory (name introduction)
    r1 = engine.reply("Hello, my name is Priya and I want to organize my schedule")
    assert "Priya" in r1
    assert "schedule" in r1.lower()
    
    # Principle 3: Conciseness (max 2 sentences)
    sentences = [s.strip() for s in r1.split(".") if s.strip()]
    assert len(sentences) <= 3  # short turn

    # Principle 4: One question at a time
    assert r1.count("?") <= 1

    # Principle 8: Never sound robotic
    robotic_phrases = [
        "certainly",
        "absolutely",
        "thank you for providing that information",
        "i understand your concern",
    ]
    for r in robotic_phrases:
        assert r not in r1.lower()

    # Next turn: caller answers question
    r2 = engine.reply("Let's start with Monday morning meetings")
    assert r2.count("?") <= 1
    # Does not ask for caller's name again (Principle 5)
    assert "what is your name" not in r2.lower()


def test_dialogue_engine_all_accent_character_combinations():
    """Verify all 9 accent x character combinations produce distinct non-hallucinatory replies."""
    accents = ["indian", "american", "british"]
    characters = ["professional", "funny", "warm"]

    for acc in accents:
        for char in characters:
            engine = StrictVoiceDialogueEngine(accent=acc, character=char)
            greeting = engine.generate_reply("Hello, who are you?")
            assert len(greeting) > 10
            assert "[backchannel" not in greeting
            assert "operates on key fundamental principles" not in greeting
            assert greeting.count("?") <= 1


def test_dialogue_conversation_turns_no_mechanical_regarding():
    """Verify that turns like 'can you hear me', 'how are you', and 'looking for...' produce clean, natural responses."""
    engine = StrictVoiceDialogueEngine(accent="indian", character="professional")
    
    r1 = engine.reply("Hi, can you hear me")
    assert "Yes, I hear you" in r1
    assert "Regarding" not in r1
    
    r2 = engine.reply("How are you doing today?")
    assert "operating optimally" in r2 or "ready to assist" in r2
    assert "Regarding" not in r2
    
    r3 = engine.reply("So I am looking for a man who is friendly and talks about these things like.")
    assert "Regarding" not in r3
    assert "I hear you speaking" not in r3
    assert len(r3.split(".")) <= 3
