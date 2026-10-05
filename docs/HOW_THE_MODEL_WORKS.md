# How NVIDIA PersonaPlex 7B Works

> A technical guide for engineers and system architects explaining the speech-to-speech architecture, conditioning tokens, and real-time streaming mechanics.

---

## 1. True Speech-to-Speech (S2S) Architecture

Traditional conversational AI platforms utilize a **cascaded 3-stage pipeline**:
$$\text{Audio} \xrightarrow{\text{STT}} \text{Text} \xrightarrow{\text{LLM}} \text{Text} \xrightarrow{\text{TTS}} \text{Audio}$$

This introduces compounding latency (typically 800ms - 1,800ms), destroys emotional vocal inflection, and makes natural conversational interruption (barge-in) unnatural.

**PersonaPlex 7B** operates as a **single, unified neural speech-to-speech model** powered by Kyutai Moshi / Mimi audio codecs:
$$\text{Audio Frames (24 kHz)} \xrightarrow{\text{Dual-Codebook Mimi}} \text{PersonaPlex 7B Transformer} \xrightarrow{\text{Dual-Decoder}} \text{Audio Frames (24 kHz)}$$

Both the user's speech and the assistant's speech flow through the same transformer simultaneously in real-time, enabling:
- Time-to-First-Audio (TTFA) under **200ms**.
- Preservation of pauses, chuckles, vocal hesitation, and prosody.
- Continuous full-duplex listening while speaking (instant acoustic interruption).

---

## 2. Prompt Formatting & Upstream Delimiters

PersonaPlex uses an exact token delimiter syntax verified against the upstream runtime:

```
<system> {prompt} <system>
```

> [!IMPORTANT]
> PersonaPlex opening and closing tags are literally `<system>`, NOT `<s>` and NOT `</system>`. Omitting this tag or using standard HTML tags causes persona conditioning failure.

### 2.1 Dynamic Context Assembly
At runtime, the gateway compiles the agent's persona prompt with:
1. Spoken persona instructions.
2. Formatted opening greeting line.
3. Call ending instructions.
4. Dynamic local time string calculated in the agent's configured timezone (e.g. `The current local time is Monday 4:15 PM in Asia/Kolkata.`).

> [!TIP]
> For a step-by-step authoring walkthrough and production templates, see the comprehensive [Prompting Guide](PROMPTING_GUIDE.md).

---

## 3. SentencePiece Token Budgets & Latency Impact

PersonaPlex conditions its KV-cache on the prompt text before processing speech frames. Prompt length directly dictates Time-to-First-Audio (TTFA) latency:

| Token Budget | TTFA Latency | Fast Start? | Recommendation |
| :--- | :--- | :--- | :--- |
| **$\le 135$ tokens** | **$< 200\text{ ms}$** | **YES** | **Recommended for production.** Immediate, natural conversational responsiveness. |
| **$136 - 150$ tokens** | $200 - 280\text{ ms}$ | Neutral | Acceptable for complex domain roles. |
| **$151 - 350$ tokens** | $280 - 450\text{ ms}$ | Slower | Amber warning in UI. Noticeable conversational lag. |
| **$> 350$ tokens** | $> 450\text{ ms}$ | NO | **Hard limit.** Gateway rejects publishing with RFC 9457 `PROMPT_TOO_LONG` error. |

---

## 4. Upstream Voice Conditioning Embeddings

PersonaPlex conditions vocal timbre, accent, pitch, and cadence through pre-computed tensor embeddings (18 official presets).

- The model receives the embedding path via the `voice_prompt` configuration.
- The 18 official presets are verified against upstream model weights (`NATF0.pt` through `NATM8.pt`).
- Free-form synthetic voice cloning is isolated from production calls to protect model reliability and legal consent.

---

## 5. Acoustic Interruption (Barge-In)

Because PersonaPlex is bidirectional, when a caller speaks while the assistant is generating audio:
1. The gateway detects speech energy ($RMS > 0.012$).
2. The gateway immediately transmits a `ControlAction.PAUSE` opcode to the worker.
3. The assistant's ongoing audio buffer is immediately silenced, allowing the caller to take the floor without awkward double-talk.
