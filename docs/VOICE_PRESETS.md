# Official PersonaPlex Voice Presets Catalog

> NVIDIA PersonaPlex 7B features 18 official upstream voice conditioning embeddings (`NATF0.pt` – `NATF8.pt` and `NATM0.pt` – `NATM8.pt`).  
> These embeddings are verified against upstream model weights and do not require runtime cloning.

---

## Complete Preset Reference Table

| Voice ID | Display Name | Gender | Accent | Speaking Style | Recommended Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `NATF0.pt` | **Natural Calm (Elena)** | Female | Neutral American | Warm, unhurried, reassuring | General assistant, clinic reception, onboarding |
| `NATF1.pt` | **Empathetic Care (Sarah)** | Female | Warm American | Gentle, attentive, patient | Healthcare, patient scheduling, mental wellness |
| `NATF2.pt` | **Professional Clear (Maya)** | Female | Mid-Atlantic | Crisp, articulate, efficient | Corporate switchboard, financial inquiries |
| `NATF3.pt` | **Friendly Upbeat (Chloe)** | Female | West Coast | Cheerful, energetic, engaging | Retail support, outbound follow-ups |
| `NATF4.pt` | **Direct Concise (Rachel)** | Female | Northeastern | Focused, confident, brisk | Tech dispatch, emergency triage |
| `NATF5.pt` | **Soft Spoken (Hannah)** | Female | Southern American | Soft, courteous, relaxed | Hospitality reservations, luxury concierge |
| `NATF6.pt` | **Dynamic Expressive (Zoe)** | Female | British RP | Expressive, polished, articulate | Educational coaching, guided walkthroughs |
| `NATF7.pt` | **Calm Measured (Grace)** | Female | Canadian | Steady, neutral, trustworthy | Insurance claims, dispute resolution |
| `NATF8.pt` | **Warm Authority (Victoria)** | Female | International | Mature, authoritative, composed | Legal reception, executive assistance |
| `NATM0.pt` | **Warm Conversational (Marcus)** | Male | Neutral American | Natural, friendly, casual | General telephone support, customer success |
| `NATM1.pt` | **Professional Direct (Alex)** | Male | Mid-Atlantic | Clear, steady, problem-solving | Technical helpdesk, IT troubleshooting |
| `NATM2.pt` | **Empathetic Support (David)** | Male | Pacific Northwest | Patient, reassuring, calm | Crisis help, patient follow-up |
| `NATM3.pt` | **Crisp Business (James)** | Male | British RP | Formal, precise, articulate | Banking verification, corporate reception |
| `NATM4.pt` | **Casual Friendly (Sam)** | Male | West Coast | Relaxed, youthful, approachable | Consumer products, e-commerce ordering |
| `NATM5.pt` | **Deep Composed (Michael)** | Male | Midwestern | Low pitch, authoritative, steady | Security alerts, system status notifications |
| `NATM6.pt` | **Brisk Efficient (Daniel)** | Male | Northeastern | Fast cadence, clear, energetic | Logistics, delivery status, courier dispatch |
| `NATM7.pt` | **Gentle Reassuring (Thomas)** | Male | Canadian | Mellow, supportive, polite | Senior care check-ins, medical reminders |
| `NATM8.pt` | **Calm Confident (Robert)** | Male | International | Experienced, measured, articulate | Professional consulting, enterprise services |

---

## Preset Verification & Invariants

1. **Zero Hallucinated Voices:** All 18 IDs map directly to concrete upstream tensor files in `models/` or the mock worker registry.
2. **Immutable Selection:** When an agent version is published, the exact voice preset ID is locked in the version record.
3. **Telephony Compatibility:** All 18 presets render with equal acoustic fidelity across 16 kHz web Linear PCM and 8 kHz G.711 μ-law telephony codecs.
