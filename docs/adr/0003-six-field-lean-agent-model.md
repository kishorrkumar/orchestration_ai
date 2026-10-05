# ADR 0003: Six-Field Lean Voice Agent Aggregate

## Context
Earlier prototypes accumulated redundant settings (cascaded fallback options, arbitrary temperature dials, unused RAG thresholds) that are incompatible with true end-to-end speech-to-speech models.

## Decision
We consolidated the Agent aggregate to strictly 6 core fields:
1. `name`: Display name & template interpolation (`{{agent_name}}`).
2. `voice_id`: One of 18 official PersonaPlex presets (`NATF0.pt` – `NATM8.pt`).
3. `greeting`: Spoken opening line with `agent_speaks_first` toggle.
4. `system_prompt`: Distraction-free persona writing canvas with live token counting.
5. `ending`: Concluding utterance for `EndOfCallDetector`.
6. `timezone_str`: Spoken time context calculator supporting half-hour offsets.

Advanced dormant features (RAG, voice cloning, cascaded fallback) were preserved and isolated in `orchestration/dormant/`.

## Consequences
- **Positive:** Clear, focused user experience modeled after human conversational design.
- **Positive:** Guaranteed token budgets for fast Time-to-First-Audio ($< 200\text{ ms}$).
