import asyncio
import time
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

def test_indian_voices():
    print("Loading Kokoro ONNX...")
    k = Kokoro('models/kokoro/kokoro-v1.0.onnx', 'models/kokoro/voices-v1.0.bin')
    indian_voices = ['hm_omega', 'hm_psi', 'hf_alpha', 'hf_beta']
    
    test_phrase = "Namaste! Haanji, absolutely, I can help you with that. Tell me what you need."
    
    for v in indian_voices:
        t0 = time.perf_counter()
        samples, sr = k.create(test_phrase, voice=v, speed=1.05, lang="en-us")
        dur = len(samples) / sr
        elapsed = (time.perf_counter() - t0) * 1000.0
        rms = np.sqrt(np.mean(samples ** 2))
        print(f"Voice: {v:<10} | Generated: {dur:.2f}s audio | Latency: {elapsed:.1f}ms | RMS: {rms:.4f}")
        sf.write(f"models/kokoro/sample_{v}.wav", samples, sr)

    # Test custom blended voice (70% hm_omega + 30% hm_psi for rich conversational male)
    style_omega = k.get_voice_style('hm_omega')
    style_psi = k.get_voice_style('hm_psi')
    blended_male = 0.65 * style_omega + 0.35 * style_psi
    t0 = time.perf_counter()
    samples, sr = k.create(test_phrase, voice=blended_male, speed=1.08, lang="en-us")
    dur = len(samples) / sr
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"Voice: blended_male | Generated: {dur:.2f}s audio | Latency: {elapsed:.1f}ms | RMS: {np.sqrt(np.mean(samples**2)):.4f}")
    sf.write("models/kokoro/sample_blended_male.wav", samples, sr)

if __name__ == "__main__":
    test_indian_voices()
