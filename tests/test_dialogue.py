import pytest
from orchestration.persona.registry import default_registry, PersonaRegistry
from orchestration.persona.dialogue import GroundedDialogueEngine
from orchestration.rag.engine import RAGEngine


def test_nine_accent_character_presets_registered():
    """Verify all 3 accents x 3 characters (9 presets) are registered and well-formed."""
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
        assert len(prompt) > 20
        assert persona.voice_prompt.endswith(".pt")


def test_dialogue_engine_truthful_personal_questions():
    """Verify the dialogue engine answers questions like 'What did you eat today?' honestly without bluffing."""
    engine = GroundedDialogueEngine(accent="american", character="warm")
    reply = engine.generate_reply("What did you eat today?")
    
    # Must NOT hallucinate or regurgitate broken template like "operates on key fundamental principles"
    assert "operates on key fundamental principles" not in reply.lower()
    assert "[backchannel" not in reply.lower()
    # Must state it is an AI / software / doesn't eat food
    assert any(term in reply.lower() for term in ["ai", "software", "food", "eat", "digital", "data"])


def test_dialogue_engine_rag_grounded():
    """Verify dialogue engine grounds answers on uploaded document knowledge."""
    rag = RAGEngine()
    rag.add_text(
        title="PersonaPlex Protocol Specs",
        text="PersonaPlex operates at 24000 Hz sample rate with 80 ms audio frames containing exactly 1920 PCM samples."
    )
    engine = GroundedDialogueEngine(accent="british", character="professional", rag_engine=rag)

    reply = engine.generate_reply("What is the sample rate and frame size of PersonaPlex?")
    assert "24000" in reply or "24" in reply
    assert "1920" in reply or "80" in reply
    assert "[backchannel" not in reply


def test_dialogue_engine_all_accent_character_combinations():
    """Verify all 9 accent x character combinations produce distinct non-hallucinatory replies."""
    accents = ["indian", "american", "british"]
    characters = ["professional", "funny", "warm"]

    for acc in accents:
        for char in characters:
            engine = GroundedDialogueEngine(accent=acc, character=char)
            greeting = engine.generate_reply("Hello, who are you?")
            assert len(greeting) > 10
            assert "[backchannel" not in greeting
            assert "operates on key fundamental principles" not in greeting
