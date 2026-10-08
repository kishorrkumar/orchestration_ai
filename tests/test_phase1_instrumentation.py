"""
Phase 1 Instrumentation & Audio Quality Verification Tests.
Tests:
- Streaming SentencePiece detokenization & apostrophe preservation
- -16 LUFS AGC speech loudness normalization
- Dual-channel 24kHz blackbox audio recorder
- Monologue guard thresholding
"""

import os
import pathlib
import tempfile
import unittest
import numpy as np
import soundfile as sf

from orchestration.audio.detokenizer import (
    clean_sentencepiece_piece,
    detokenize_sentencepiece_stream,
    stitch_token,
)
from orchestration.audio.dsp import compute_rms, normalize_speech_loudness, soft_clip
from orchestration.audio.recorder import SessionAudioRecorder


class TestPhase1Instrumentation(unittest.TestCase):

    def test_detokenizer_preserves_contractions_and_apostrophes(self):
        """Verify that SentencePiece tokens are detokenized without losing apostrophes or mangling words."""
        # Case 1: "I'm"
        tokens1 = [" I", "'", "m", " doing", " great"]
        result1 = detokenize_sentencepiece_stream(tokens1)
        self.assertEqual(result1, "I'm doing great")

        # Case 2: "you're" and "we'll"
        tokens2 = [" you", "'", "re", " busy", " but", " we", "'", "ll", " follow", " up"]
        result2 = detokenize_sentencepiece_stream(tokens2)
        self.assertEqual(result2, "you're busy but we'll follow up")

        # Case 3: ASR recovery pattern e.g. "I' doing" -> "I'm doing"
        tokens3 = ["I'", " doing", " well"]
        result3 = detokenize_sentencepiece_stream(tokens3)
        self.assertEqual(result3, "I'm doing well")

        # Case 4: Punctuation attachment
        tokens4 = ["Hello", ",", " this", " is", " Snapserve", "!"]
        result4 = detokenize_sentencepiece_stream(tokens4)
        self.assertEqual(result4, "Hello, this is Snapserve!")

    def test_speech_loudness_normalization_boosts_quiet_speech(self):
        """Verify AGC amplifies quiet speech (RMS 0.0005) up towards target 0.12 without harsh clipping."""
        t = np.linspace(0, 0.08, 1920, endpoint=False, dtype=np.float32)
        quiet_signal = (0.0007 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        initial_rms = compute_rms(quiet_signal)
        self.assertLess(initial_rms, 0.001)

        # Normalize speech loudness
        normalized = normalize_speech_loudness(quiet_signal, target_rms=0.12, max_gain_factor=12.0)
        norm_rms = compute_rms(normalized)

        # Must be boosted significantly by max_gain_factor
        self.assertGreaterEqual(norm_rms, initial_rms * 8.0)
        self.assertLessEqual(np.max(np.abs(normalized)), 1.0)

    def test_speech_loudness_normalization_leaves_silence_alone(self):
        """Verify AGC does not boost pure noise floor or silence below min_speech_rms."""
        silence = np.full(1920, 0.00005, dtype=np.float32)
        normalized = normalize_speech_loudness(silence, target_rms=0.12, min_speech_rms=0.0008)
        np.testing.assert_array_almost_equal(silence, normalized)

    def test_session_audio_recorder_dual_channel(self):
        """Verify SessionAudioRecorder creates a 24 kHz stereo WAV with aligned channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = pathlib.Path(tmpdir)
            import orchestration.audio.recorder as rec_mod
            old_dir = rec_mod.RECORDINGS_DIR
            rec_mod.RECORDINGS_DIR = rec_dir

            try:
                recorder = SessionAudioRecorder("test_session_123", sample_rate=24000, enabled=True)
                # Channel 0: Caller (inbound)
                in_chunk = np.ones(2400, dtype=np.float32) * 0.1
                recorder.record_inbound(in_chunk)

                # Channel 1: Assistant (outbound)
                out_chunk = np.ones(4800, dtype=np.float32) * 0.2
                recorder.record_outbound(out_chunk)

                summary = recorder.close()
                self.assertTrue(summary["enabled"])
                self.assertTrue(summary["recorded"])
                self.assertEqual(summary["channels"], 2)

                file_path = pathlib.Path(summary["file_path"])
                self.assertTrue(file_path.exists())

                data, sr = sf.read(str(file_path))
                self.assertEqual(sr, 24000)
                self.assertEqual(data.ndim, 2)
                self.assertEqual(data.shape[1], 2)
                self.assertEqual(len(data), 4800)
            finally:
                rec_mod.RECORDINGS_DIR = old_dir

    def test_voice_cloning_speaker_similarity_qa(self):
        """Verify compute_speaker_similarity yields >= 0.75 for identical/matched voice audio."""
        from orchestration.audio.similarity import compute_speaker_similarity

        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # Voice A harmonic signal
        voice_a = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # Voice A similar speaker with conversational vocal modulation
        voice_a_similar = (voice_a * (0.8 + 0.2 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        # Voice B very different pitch/timbre (880 Hz)
        voice_b = (0.5 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.sin(2 * np.pi * 1760 * t)).astype(np.float32)

        sim_matched = compute_speaker_similarity(voice_a, voice_a_similar)
        sim_unmatched = compute_speaker_similarity(voice_a, voice_b)

        self.assertGreaterEqual(sim_matched, 0.75, f"Matched similarity too low: {sim_matched}")
        self.assertLess(sim_unmatched, sim_matched, "Unmatched similarity should be lower than matched")


if __name__ == "__main__":
    unittest.main()
