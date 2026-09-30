"""
Objective Speaker-Embedding Similarity Metric for Voice Cloning Validation.

Computes cosine similarity between reference speaker audio and synthesized
audio output to verify acoustic fidelity and prevent speaker drift.

Supports:
1. Resemblyzer (if installed with d-vector)
2. Torchaudio / PyTorch acoustic filterbank feature encoder (portable fallback)
"""

from __future__ import annotations
import io
import logging
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SIMILARITY_PASS_THRESHOLD = 0.65  # Documented threshold for voice cloning similarity


def _load_audio_to_tensor(audio: Union[bytes, np.ndarray, torch.Tensor], target_sr: int = 16000) -> torch.Tensor:
    """Load audio input into a 1D float32 PyTorch tensor at target_sr."""
    if isinstance(audio, bytes):
        with io.BytesIO(audio) as bio:
            data, sr = sf.read(bio, dtype="float32")
    elif isinstance(audio, np.ndarray):
        data = audio.astype(np.float32)
        sr = 24000  # Default PersonaPlex rate if array passed
    elif isinstance(audio, torch.Tensor):
        data = audio.detach().cpu().numpy().astype(np.float32)
        sr = 24000
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    if data.ndim > 1:
        data = np.mean(data, axis=-1)

    # Simple resample to 16kHz for speaker embedding if needed
    if sr != target_sr:
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)

    tensor = torch.from_numpy(data)
    # Normalize peak amplitude
    max_val = torch.max(torch.abs(tensor))
    if max_val > 1e-6:
        tensor = tensor / max_val
    return tensor


def compute_spectral_speaker_embedding(waveform: torch.Tensor, n_mels: int = 64) -> torch.Tensor:
    """
    Compute a fixed-dimensional speaker acoustic representation:
    Mel-spectrogram + delta features + statistical pooling (mean + std).
    Produces a 256-dimensional speaker fingerprint without requiring external weights.
    """
    if len(waveform) < 800:
        return torch.zeros(n_mels * 4)

    # 1. Compute STFT spectrogram
    window = torch.hann_window(400)
    stft = torch.stft(
        waveform,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        return_complex=True
    )
    mag = torch.abs(stft)  # (freq_bins, frames)

    # 2. Simple triangular filterbank down to n_mels
    freq_bins, frames = mag.shape
    step = freq_bins / (n_mels + 2)
    filters = torch.zeros(n_mels, freq_bins)
    for i in range(n_mels):
        left = int(i * step)
        center = int((i + 1) * step)
        right = int((i + 2) * step)
        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, min(right, freq_bins)):
            if right > center:
                filters[i, j] = (right - j) / (right - center)

    mel_spec = torch.matmul(filters, mag) + 1e-6
    log_mel = torch.log(mel_spec)  # (n_mels, frames)

    # 3. Delta features
    if frames > 2:
        deltas = log_mel[:, 1:] - log_mel[:, :-1]
        deltas = F.pad(deltas, (0, 1), mode="replicate")
    else:
        deltas = torch.zeros_like(log_mel)

    # 4. Statistical pooling across time (mean + std)
    mean_mel = torch.mean(log_mel, dim=1)
    std_mel = torch.std(log_mel, dim=1)
    mean_delta = torch.mean(deltas, dim=1)
    std_delta = torch.std(deltas, dim=1)

    # Concatenate into 256-dimensional vector
    embedding = torch.cat([mean_mel, std_mel, mean_delta, std_delta], dim=0)
    # L2 normalize
    norm = torch.norm(embedding, p=2)
    if norm > 1e-6:
        embedding = embedding / norm
    return embedding


def compute_speaker_similarity(
    reference_audio: Union[bytes, np.ndarray, torch.Tensor],
    generated_audio: Union[bytes, np.ndarray, torch.Tensor],
) -> float:
    """
    Computes cosine similarity between reference speaker audio and generated voice output.
    Returns float score in [-1.0, 1.0]. Values >= 0.65 indicate high speaker timbre retention.
    """
    # 1. Try Resemblyzer if installed
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        if isinstance(reference_audio, bytes):
            with io.BytesIO(reference_audio) as bio:
                ref_wav, _ = sf.read(bio)
        else:
            ref_wav = np.asarray(reference_audio)
        if isinstance(generated_audio, bytes):
            with io.BytesIO(generated_audio) as bio:
                gen_wav, _ = sf.read(bio)
        else:
            gen_wav = np.asarray(generated_audio)

        emb_ref = encoder.embed_utterance(preprocess_wav(ref_wav))
        emb_gen = encoder.embed_utterance(preprocess_wav(gen_wav))
        sim = float(np.dot(emb_ref, emb_gen) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_gen)))
        return round(sim, 4)
    except Exception:
        pass

    # 2. Robust statistical mel-spectral speaker embedding fallback
    t_ref = _load_audio_to_tensor(reference_audio)
    t_gen = _load_audio_to_tensor(generated_audio)

    emb_ref = compute_spectral_speaker_embedding(t_ref)
    emb_gen = compute_spectral_speaker_embedding(t_gen)

    cosine_sim = float(F.cosine_similarity(emb_ref.unsqueeze(0), emb_gen.unsqueeze(0)).item())
    return round(float(np.clip(cosine_sim, -1.0, 1.0)), 4)
