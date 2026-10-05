# PersonaPlex 7B Prompting Guide

> A practical, comprehensive guide to crafting high-quality, ultra-low latency voice agent prompts for the **NVIDIA PersonaPlex 7B** Speech-to-Speech (S2S) model.

---

## 1. Speech-to-Speech (S2S) Prompting vs. Text LLM Prompting

Prompting a real-time full-duplex speech model like PersonaPlex 7B is fundamentally different from prompting text LLMs (like ChatGPT, Claude, or GPT-4):

| Dimension | Text LLMs (ChatGPT / Claude) | PersonaPlex 7B (Speech-to-Speech) |
| :--- | :--- | :--- |
| **Output Modality** | Generates text tokens displayed on screen. | **Directly generates spoken audio waveforms (24 kHz) and intonation.** |
| **Punctuation Impact** | Visual structure for reading. | **Dictates breath, cadence, pausing, and vocal emphasis.** |
| **Formatting** | Bullets, markdown headings, bold text, tables. | **Destructive.** Markdown symbols, asterisks, and numbering degrade spoken prosody. |
| **Negative Constraints** | Long lists of "NEVER / DO NOT" work adequately. | S2S models struggle with negative rule lists. **Positive conversational style guidance works best.** |
| **Token Budget & Latency** | Thousands of prompt tokens cached at nominal cost. | **Strictly sensitive to prompt length.** Keeping prompt $\le 135$ tokens ensures $<200\text{ ms}$ latency. |
| **Language Support** | Multilingual (dozens of languages). | **English only.** Non-English text causes phonetic hallucinations. |

---

## 2. The 6 Lean Agent Fields

Every voice agent in the platform consists of exactly **6 core fields**:

```
┌────────────────────────────────────────────────────────┐
│ 1. Name:            Display name & {{agent_name}}      │
│ 2. Voice:           Upstream voice preset (1 of 18)   │
│ 3. Greeting:        Opening spoken line                │
│ 4. System Prompt:   Persona identity, style & goal     │
│ 5. Ending Text:     Natural wrap-up closure line       │
│ 6. Timezone:        IANA timezone (e.g. Asia/Kolkata)  │
└────────────────────────────────────────────────────────┘
```

### Field Breakdown

1. **Name** (`name`):
   - Example: `Sarah from Horizon Dental`
   - Automatically available as `{{agent_name}}` in prompts and greetings.
2. **Voice** (`voice_id`):
   - One of 18 official upstream voice conditioning embeddings (`NATF0.pt` - `NATF8.pt`, `NATM0.pt` - `NATM8.pt`).
   - Match the vocal timbre and gender to your persona role (see [VOICE_PRESETS.md](VOICE_PRESETS.md)).
3. **Greeting** (`greeting_text`):
   - The exact opening phrase spoken by the agent when the call connects.
   - Example: `Hi there, thanks for calling Horizon Dental! This is Sarah. How can I help you today?`
4. **System Prompt** (`system_prompt`):
   - Defines the agent's identity, conversational tone, primary goal, and scope boundaries.
   - Must be written in natural spoken English prose (no markdown, no bullets).
5. **Ending Text** (`ending_text`):
   - The closing phrase the agent speaks when wrapping up the call.
   - Monitored by the supervisory `EndOfCallDetector` state machine to cleanly trigger automatic call hangup.
   - Example: `Thanks for calling Horizon Dental. Have a wonderful day, goodbye!`
6. **Timezone** (`timezone`):
   - An IANA timezone identifier (e.g., `America/New_York`, `Europe/London`, `Asia/Kolkata`).
   - The system automatically injects the caller's local weekday, time, and day-part into the model context.

---

## 3. Template Variables & Dynamic Temporal Context

The prompt compiler dynamically interpolates variables enclosed in double curly braces `{{variable}}`:

### Built-in Dynamic Variables

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `{{agent_name}}` | Name of the voice agent | `Sarah` |
| `{{current_time}}` | Local time formatted without leading zero | `3:15 PM` |
| `{{weekday}}` | Current day of the week | `Monday` |
| `{{date}}` | Formatted date | `October 05, 2026` |
| `{{day_part}}` | Time-of-day category: `morning`, `afternoon`, `evening`, `night` | `afternoon` |
| `{{caller_name}}` | Name of the inbound/outbound caller (if supplied) | `Alex` |
| `{{customer_name}}` | Customer company or account name (if supplied) | `Acme Corp` |
| `{{phone_number}}` | Caller's telephone number | `+1-555-0199` |

### Automatic Local Time Line
At the start of every call session, the gateway calculates the exact local time in the agent's timezone and automatically injects this context line into the prompt:
```
It is Monday, 3:15 PM (afternoon) for the caller.
```
This enables the agent to naturally use appropriate temporal greetings (e.g., *"Good afternoon!"* on a Monday at 3 PM) without hallucinating time.

---

## 4. How the Gateway Assembles the System Prompt

The prompt compiler combines your inputs into a single, compact body and wraps it with the verified upstream PersonaPlex delimiters:

```
<system>
{system_prompt}
It is {weekday}, {current_time} ({day_part}) for the caller.
Start: Open the call by saying: "{greeting_text}"
Close: When the conversation is done, say: "{ending_text}"
<system>
```

> [!IMPORTANT]
> Both opening and closing delimiters are literally `<system>`, NOT `<s>` and NOT `</system>`. The gateway manages this wrapping automatically.

---

## 5. The Golden Rules of Voice Prompting

### DO:
1. **Write for the ear, not the eye**:
   - Read your prompt aloud. If it sounds stiff or like a written essay, rewrite it into natural spoken dialogue.
2. **Use short, conversational sentences**:
   - Limit responses to 1–2 brief sentences per conversational turn. Long monologues cause awkward user interruption.
3. **Use natural contractions**:
   - Write *"I'm"*, *"we'll"*, *"don't"*, *"let's"*, *"you're"* instead of *"I am"*, *"we will"*, *"do not"*.
4. **Phonetic and plain-text numbers**:
   - Write *"twenty-five dollars"* or *"$25"* instead of *"USD 25.00"*.
   - Write *"nine A M"* or *"9 AM"* instead of *"09:00:00 UTC"*.
5. **State the agent's primary goal clearly**:
   - Explicitly define the purpose: *"Your goal is to answer questions about menu items and take table reservations."*
6. **Keep it under 135 tokens**:
   - Keep the prompt lean and focused. Prompt length directly determines Time-to-First-Audio (TTFA) latency.

### DON'T:
1. **Never use Markdown formatting**:
   - **No** `**bold**` or `*italics*`.
   - **No** `# Heading 1` or `## Heading 2`.
   - **No** `- bullet points` or `1. numbered lists`.
   - *Why:* S2S audio decoders may vocalize symbols (e.g. saying *"star star bold star star"*) or introduce unnatural pauses.
2. **Never include Emojis**:
   - **No** 😊, 🚀, 👍, 📞.
   - *Why:* The acoustic tokenizer will attempt to pronounce the unicode symbol or produce garbled noise.
3. **Avoid long lists of "NEVER / ALWAYS / DO NOT" rules**:
   - Avoid: *"NEVER say this, NEVER do that, DO NOT answer if..."*
   - Prefer: *"Speak warmly and concisely. Focus strictly on appointment scheduling. If asked about billing, politely redirect them to our web portal."*
4. **Never use non-English characters**:
   - PersonaPlex 7B weights are strictly trained on English speech. Accented characters or foreign scripts cause phonetic hallucinations.
5. **Don't dump entire knowledge bases into the prompt**:
   - Detailed manuals or long policy documents blow past the 150-token budget and degrade conversational latency. Keep the prompt focused on persona and conversational guardrails.

---

## 6. SentencePiece Token Budgets & Latency Guide

Prompt token counts are computed using the official 32k PersonaPlex SentencePiece model (`models/tokenizer_spm_32k_3.model`):

```
0 ──────────── 135 ────────── 150 ────────────────────── 350 ─────────► Tokens
  [ Ideal / Fast Start ]     [ Acceptable ]    [ Amber Warning ]    [ Blocked ]
      < 200ms TTFA            200-280ms           280-450ms           > 450ms
```

| Token Range | Status | Studio Badge | TTFA Latency | Description |
| :--- | :--- | :--- | :--- | :--- |
| **$\le 135$ tokens** | **Ideal** | Green badge: `Fast Start (<200ms)` | **$< 200\text{ ms}$** | Recommended for all production agents. Crisp, immediate conversational responses. |
| **$136 - 150$ tokens** | **Acceptable** | Green badge: `Fast Start` | **$200 - 280\text{ ms}$** | Standard latency. Suitable for slightly more complex persona instructions. |
| **$151 - 350$ tokens** | **Warning** | Amber badge: `Above Budget` | **$280 - 450\text{ ms}$** | Noticeable conversational hesitation on initial connect. Publishable, but triggers a linter warning. |
| **$> 350$ tokens** | **Blocked** | Red badge: `Exceeds Limit` | **$> 450\text{ ms}$** | **Publish blocked.** Rejected by the gateway with RFC 9457 `PROMPT_TOO_LONG`. |

---

## 7. Production Prompt Templates (Ready to Copy-Paste)

### Template 1: Clinic & Healthcare Appointment Scheduler
- **Agent Name**: `Sarah from Apex Clinic`
- **Voice Preset**: `NATF2.pt` (Warm, professional female)
- **Timezone**: `America/New_York`
- **Greeting**: `Hi, thank you for calling Apex Medical Clinic! My name is Sarah. Are you looking to schedule or reschedule an appointment today?`
- **Ending Text**: `Thank you for calling Apex Medical Clinic. Have a wonderful day, goodbye!`
- **System Prompt**:
```text
You are Sarah, a warm and helpful medical receptionist at Apex Medical Clinic. Your goal is to help callers book or reschedule checkups and routine doctor visits. Speak warmly, speak in 1 to 2 short sentences per turn, and confirm the patient's preferred day and time. If the caller describes an emergency, immediately advise them to hang up and dial 911.
```
- **Token Breakdown**:
  - System prompt: ~62 tokens
  - Greeting + Ending: ~44 tokens
  - Total compiled: **~122 tokens** (🟢 *Fast Start*)

---

### Template 2: Friendly Customer Support & Order Triage
- **Agent Name**: `Alex Support`
- **Voice Preset**: `NATM0.pt` (Direct, calm male)
- **Timezone**: `America/Los_Angeles`
- **Greeting**: `Hey there, thanks for calling customer care! I'm Alex. What can I help you with today?`
- **Ending Text**: `Glad I could help. Have a great day, goodbye!`
- **System Prompt**:
```text
You are Alex, an efficient and cheerful customer support specialist. Your goal is to assist customers with order tracking, returns, and delivery updates. Keep answers brief and conversational, usually 1 or 2 sentences. Always ask for the order number if not provided.
```
- **Token Breakdown**:
  - System prompt: ~46 tokens
  - Greeting + Ending: ~36 tokens
  - Total compiled: **~98 tokens** (🟢 *Fast Start*)

---

### Template 3: Inbound Sales & Product Qualification (SDR)
- **Agent Name**: `Jordan from CloudScale`
- **Voice Preset**: `NATM2.pt` (Polished, confident male)
- **Timezone**: `America/Chicago`
- **Greeting**: `Hello, thanks for reaching out to CloudScale! My name is Jordan. Are you exploring our platform for your team or a current project?`
- **Ending Text**: `Thanks for your time today. Our team will follow up shortly. Take care, goodbye!`
- **System Prompt**:
```text
You are Jordan, an energetic and knowledgeable sales specialist at CloudScale. Your goal is to understand the caller's cloud infrastructure requirements and team size, answer high-level questions, and schedule a 15-minute product demonstration. Be curious, positive, and keep your turns brief and conversational.
```
- **Token Breakdown**:
  - System prompt: ~56 tokens
  - Greeting + Ending: ~43 tokens
  - Total compiled: **~115 tokens** (🟢 *Fast Start*)

---

### Template 4: IT Helpdesk & Password Reset Specialist
- **Agent Name**: `Elena from IT Services`
- **Voice Preset**: `NATF0.pt` (Crisp, clear female)
- **Timezone**: `Europe/London`
- **Greeting**: `Hi, you've reached the internal IT Helpdesk. This is Elena. Are you calling regarding a password reset or system access?`
- **Ending Text**: `Your ticket has been updated. Thanks for calling IT, goodbye!`
- **System Prompt**:
```text
You are Elena, a calm and methodical IT helpdesk specialist. Your goal is to troubleshoot workstation login issues, VPN connectivity, and initiate self-service password resets. Give step-by-step instructions one sentence at a time, and wait for the caller to confirm before continuing.
```
- **Token Breakdown**:
  - System prompt: ~54 tokens
  - Greeting + Ending: ~39 tokens
  - Total compiled: **~109 tokens** (🟢 *Fast Start*)

---

### Template 5: Boutique Hotel Concierge & Dining Reservations
- **Agent Name**: `Claire from The Grandview`
- **Voice Preset**: `NATF4.pt` (Sophisticated, friendly female)
- **Timezone**: `Europe/Paris`
- **Greeting**: `Good day and welcome to The Grandview Hotel. My name is Claire. How may I assist with your stay or dinner reservations today?`
- **Ending Text**: `We look forward to welcoming you to The Grandview. Have a lovely day, goodbye!`
- **System Prompt**:
```text
You are Claire, a polite and attentive concierge at The Grandview Hotel. Your goal is to assist guests with rooftop dinner reservations, spa appointments, and local city recommendations. Speak with courteous warmth, answer in 1 or 2 elegant sentences, and confirm guest party sizes.
```
- **Token Breakdown**:
  - System prompt: ~58 tokens
  - Greeting + Ending: ~42 tokens
  - Total compiled: **~116 tokens** (🟢 *Fast Start*)

---

## 8. How to Create and Test an Agent

### Method 1: Using the Web Studio UI
1. Navigate to `http://127.0.0.1:8000/` in your browser.
2. Click **New Agent** (top right).
3. Fill in the **Name**, select a **Voice**, configure **Greeting** and **System Prompt**, and set your **Ending**.
4. Observe the live **Token Counter**:
   - 🟢 Green badge indicates **$\le 135$ tokens (Fast Start)**.
   - Any warnings (e.g., markdown, emojis, excess rules) appear immediately in the lint panel.
5. Click **Save & Publish** (`Ctrl + Enter`).
6. Click **Test Call** (`T`) to open the in-browser 16 kHz voice modal and talk directly to your agent.

### Method 2: Using the REST API
You can programmatically create and publish agents using the REST endpoints:

```bash
# 1. Create a draft agent
curl -X POST http://127.0.0.1:8000/v2/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sarah from Apex Clinic",
    "voice_id": "NATF2.pt",
    "greeting_text": "Hi, thank you for calling Apex Clinic! How can I help you?",
    "system_prompt": "You are Sarah, a warm medical receptionist. Help callers schedule appointments.",
    "ending_text": "Thank you for calling. Have a wonderful day, goodbye!",
    "timezone": "America/New_York"
  }'

# 2. Publish to snapshot version 1
curl -X POST http://127.0.0.1:8000/v2/agents/{agent_id}/publish \
  -H "Content-Type: application/json" \
  -d '{"change_note": "Initial production release"}'
```

### Method 3: Connecting via Voice WebSocket
```
ws://127.0.0.1:8000/v2/voice?agent_id={agent_id}&sample_rate=16000&codec=pcm16
```
- **Stream In**: 16 kHz 16-bit mono PCM chunks (or 8 kHz G.711 $\mu$-law for telephony).
- **Stream Out**: 16 kHz 16-bit mono PCM chunks + JSON transcript events (`{"type": "transcript", "text": "..."}`).
- **Automatic Hangup**: The supervisory `EndOfCallDetector` monitors the closing phrase and clean silence window to emit `{"type": "call_ended", "reason": "agent_closed"}`.

---

## 9. Troubleshooting & FAQ

### Q: Why does the agent hesitate for 1-2 seconds before speaking?
**A:** Check your token count. If your prompt exceeds 150 tokens, the initial KV-cache conditioning handshake takes longer. Reduce your prompt to $\le 135$ tokens for immediate $<200$ms response times.

### Q: Why did the agent vocalize strange characters?
**A:** Check for markdown formatting (`**`, `##`, `-`), bullet dashes, or emojis. S2S neural decoders treat all symbols as text to be phonetically spoken. Remove all non-conversational formatting.

### Q: Why didn't the call hang up automatically when the agent said goodbye?
**A:** Ensure your `ending_text` in the agent configuration closely matches what the agent actually said. The `EndOfCallDetector` uses fuzzy matching ($\ge 80\%$ similarity) against the configured ending phrase, followed by a 1.5-second quiet window to allow the audio buffer to drain. If the user speaks during that quiet window, the hangup is canceled to allow the conversation to continue.

### Q: Can I use languages other than English?
**A:** No. The NVIDIA PersonaPlex 7B foundation model weights are trained on English audio and text corpora. Inputting foreign languages will produce unintelligible phonetic speech.
