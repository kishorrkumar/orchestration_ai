from orchestration.persona.registry import (
    OFFICIAL_VOICE_PRESETS,
    PersonaConfig,
    PersonaRegistry,
    wrap_with_system_tags,
    normalize_voice_name,
)


def test_system_tag_wrapping():
    raw = "You are a helpful assistant."
    wrapped = wrap_with_system_tags(raw)
    assert wrapped == "<system> You are a helpful assistant. <system>"

    # Idempotent if already wrapped
    assert wrap_with_system_tags(wrapped) == "<system> You are a helpful assistant. <system>"


def test_normalize_voice_name():
    assert normalize_voice_name("NATF2") == "NATF2.pt"
    assert normalize_voice_name("NATF2.pt") == "NATF2.pt"
    assert normalize_voice_name("custom_voice.wav") == "custom_voice.wav"


def test_official_presets_count():
    assert len(OFFICIAL_VOICE_PRESETS) == 18
    # 4 NATF, 4 NATM, 5 VARF, 5 VARM
    natf = [v for v in OFFICIAL_VOICE_PRESETS if v.startswith("NATF")]
    natm = [v for v in OFFICIAL_VOICE_PRESETS if v.startswith("NATM")]
    varf = [v for v in OFFICIAL_VOICE_PRESETS if v.startswith("VARF")]
    varm = [v for v in OFFICIAL_VOICE_PRESETS if v.startswith("VARM")]
    assert len(natf) == 4
    assert len(natm) == 4
    assert len(varf) == 5
    assert len(varm) == 5


def test_persona_registry_defaults():
    registry = PersonaRegistry()
    personas = registry.list_all()
    assert len(personas) == 2

    aarav = registry.get("indian_pro")
    assert aarav is not None
    assert aarav.voice_prompt == "NATM0.pt"
    assert "en-IN-PrabhatNeural" in aarav.neural_voice
    assert aarav.get_formatted_text_prompt().startswith("<system>")
    assert aarav.get_formatted_text_prompt().endswith("<system>")

    rohan = registry.get("indian_funny")
    assert rohan is not None
    assert "en-IN" in rohan.neural_voice


def test_custom_persona_registration():
    registry = PersonaRegistry()
    custom = PersonaConfig(
        id="tech_support",
        name="Support Bot",
        voice_prompt="NATM3",
        text_prompt="Help users debug networking issues.",
    )
    registry.register(custom)
    retrieved = registry.get("tech_support")
    assert retrieved is not None
    assert retrieved.get_normalized_voice_prompt() == "NATM3.pt"
    assert "<system> Help users debug networking issues. <system>" in retrieved.get_formatted_text_prompt()

    assert registry.delete("tech_support") is True
    assert registry.get("tech_support") is None
