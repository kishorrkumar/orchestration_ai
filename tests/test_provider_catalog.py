"""
Tests for Provider Catalog manifests and Pydantic validation.
"""

from orchestration.domain.providers.catalog import ProviderCatalog, ProviderKind


def test_catalog_loads_all_manifests():
    catalog = ProviderCatalog()
    providers = catalog.list_all()
    assert len(providers) >= 11

    # Check key provider IDs
    provider_ids = {p.id for p in providers}
    expected_ids = {
        "deepgram_stt",
        "sarvam_stt",
        "openai_llm",
        "anthropic_llm",
        "cartesia_tts",
        "elevenlabs_tts",
        "deepgram_tts",
        "sarvam_tts",
        "fake_stt",
        "fake_llm",
        "fake_tts",
    }
    assert expected_ids.issubset(provider_ids)


def test_catalog_grouping_by_kind():
    catalog = ProviderCatalog()
    stt_providers = catalog.list_by_kind(ProviderKind.STT)
    llm_providers = catalog.list_by_kind(ProviderKind.LLM)
    tts_providers = catalog.list_by_kind(ProviderKind.TTS)

    assert len(stt_providers) >= 3  # Deepgram, Sarvam, Fake
    assert len(llm_providers) >= 3  # OpenAI, Anthropic, Fake
    assert len(tts_providers) >= 5  # Cartesia, ElevenLabs, Deepgram Aura, Sarvam, Fake


def test_catalog_native_eot_flags():
    catalog = ProviderCatalog()
    deepgram = catalog.get("deepgram_stt")
    assert deepgram is not None
    assert deepgram.native_eot is True

    sarvam = catalog.get("sarvam_stt")
    assert sarvam is not None
    assert sarvam.native_eot is False


def test_catalog_fake_providers_have_defaults():
    catalog = ProviderCatalog()
    fake_stt = catalog.get("fake_stt")
    fake_llm = catalog.get("fake_llm")
    fake_tts = catalog.get("fake_tts")

    assert fake_stt is not None and fake_stt.default_model == "mock-flux"
    assert fake_llm is not None and fake_llm.default_model == "mock-gpt"
    assert fake_tts is not None and fake_tts.default_voice == "mock-alex"
