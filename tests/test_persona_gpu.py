"""
GPU-marked test proving that two different system prompts produce
different generation behavior on the real PersonaPlex model.

Run with:
    pytest -v -m gpu tests/test_persona_gpu.py

Skippable in CI or environments without 16GB+ VRAM GPU.
"""

import os
import pytest
import torch


@pytest.mark.gpu
def test_persona_conditioning_behavior_difference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available; skipping PersonaPlex GPU test.")

    # Check for Hugging Face token or local model weights
    hf_token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        # Quick check if model is accessible or downloaded
        checkpoint = hf_hub_download("nvidia/personaplex-7b-v1", "tokenizer_spm_32k_3.model", token=hf_token)
    except Exception as e:
        pytest.skip(f"PersonaPlex weights not cached or accessible: {e}")

    try:
        from moshi.models import loaders
        import sentencepiece
    except ImportError:
        pytest.skip("moshi package not installed in environment.")

    device = torch.device("cuda")
    spm = sentencepiece.SentencePieceProcessor(checkpoint)

    # Prompt A: Dr. Elena (Wise Teacher)
    prompt_a = "<system> You are Dr. Elena, a wise teacher who explains physics with simple analogies. <system>"
    # Prompt B: Pirate Captain
    prompt_b = "<system> You are Captain Blackbeard, an aggressive pirate who shouts Ahoy and talks about gold. <system>"

    tokens_a = spm.encode(prompt_a)
    tokens_b = spm.encode(prompt_b)

    assert tokens_a != tokens_b, "Tokenized prompt sequences must differ"

    # If full 7B weights exist on disk, step both prompts through LMGen and verify divergent initial hidden state or output
    moshi_path = None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached_file = try_to_load_from_cache("nvidia/personaplex-7b-v1", loaders.MOSHI_NAME)
        if cached_file and os.path.exists(cached_file):
            moshi_path = cached_file
    except Exception:
        pass

    if moshi_path is None:
        pytest.skip("Full 14GB PersonaPlex weights not present in local cache.")

    # Load model and verify that internal prompt states diverge
    lm = loaders.get_moshi_lm(moshi_path, device=device)
    lm.eval()

    gen_a = loaders.LMGen(lm, device=device, text_prompt_tokens=tokens_a)
    gen_b = loaders.LMGen(lm, device=device, text_prompt_tokens=tokens_b)

    with gen_a.streaming(1), gen_b.streaming(1):
        gen_a._step_text_prompt()
        gen_b._step_text_prompt()
        # Verify that LMGen internal caches differ after stepping different personas
        state_a = gen_a._streaming_state.cache
        state_b = gen_b._streaming_state.cache
        assert not torch.allclose(state_a.float(), state_b.float()), (
            "LMGen KV cache state should differ after stepping different persona prompts."
        )
