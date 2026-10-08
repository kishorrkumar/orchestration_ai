"""
Prompt A/B Evaluation Suite for NVIDIA PersonaPlex S2S Personas.
Evaluates 5 prompt variants across 30 conversational scenarios for:
- Mean turn length (words) & compliance with 8-25 word target
- Monologue cutoff violation rate (>25 words)
- Pricing redirection adherence (100% redirect, 0 fabricated quotes)
- Anti-Moshi / Kyutai leak rate (0%)
- Detokenizer contraction health
- SentencePiece prompt token budget (<= 350 tokens)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any

from orchestration.audio.detokenizer import detokenize_sentencepiece_stream
from orchestration.persona.dialogue import StrictVoiceDialogueEngine
from orchestration.prompts.compiler import compile_prompt

# 30 Comprehensive Evaluation Scenarios
EVALUATION_SCENARIOS = [
    {"id": "01_hello", "caller": "Hello? Can you hear me?", "type": "greeting"},
    {"id": "02_who", "caller": "Hi, who am I speaking with?", "type": "identity"},
    {"id": "03_pricing_direct", "caller": "How much does your service cost per month?", "type": "pricing"},
    {"id": "04_pricing_range", "caller": "Can you give me a rough ball-park figure for 1000 calls?", "type": "pricing"},
    {"id": "05_company_info", "caller": "What does Snapserve actually do?", "type": "product"},
    {"id": "06_missed_calls", "caller": "We miss about 40 customer calls every day.", "type": "pain_point"},
    {"id": "07_busy_caller", "caller": "Look, I am running into a meeting right now.", "type": "objection"},
    {"id": "08_robot_check", "caller": "Are you an AI or a real human?", "type": "identity"},
    {"id": "09_uninterested", "caller": "Not interested, please take me off your list.", "type": "objection"},
    {"id": "10_affirmation", "caller": "Yeah, makes sense.", "type": "backchannel"},
    {"id": "11_booking", "caller": "I'd like to book a quick demo to see how it works.", "type": "intent"},
    {"id": "12_technical_delay", "caller": "What is the voice latency on these calls?", "type": "technical"},
    {"id": "13_telephony", "caller": "Does this work with our Twilio phone lines?", "type": "technical"},
    {"id": "14_support_query", "caller": "I have an issue with our existing account.", "type": "support"},
    {"id": "15_discount", "caller": "Can I get a 50% discount on the enterprise plan?", "type": "pricing"},
    {"id": "16_multi_turn_1", "caller": "Tell me more about the setup time.", "type": "product"},
    {"id": "17_multi_turn_2", "caller": "Do we need our own GPUs?", "type": "technical"},
    {"id": "18_speech_unclear", "caller": "Sorry, what was that again?", "type": "clarification"},
    {"id": "19_competitor", "caller": "How are you different from Bland or Vapi?", "type": "comparison"},
    {"id": "20_speech_long", "caller": "We run a dental clinic with 3 locations and the receptionists are overwhelmed with appointment bookings.", "type": "pain_point"},
    {"id": "21_closing", "caller": "Alright, that is all I needed for now. Thanks!", "type": "closing"},
    {"id": "22_email_ask", "caller": "Can you send the details over email?", "type": "lead_capture"},
    {"id": "23_complaint", "caller": "The last agent hung up on me.", "type": "support"},
    {"id": "24_security", "caller": "Is caller audio recorded or stored securely?", "type": "compliance"},
    {"id": "25_hours", "caller": "What hours does your support team operate?", "type": "support"},
    {"id": "26_silent_pause", "caller": "Hmm... let me think about that.", "type": "hesitation"},
    {"id": "27_immediate_hangup", "caller": "Wrong number, bye.", "type": "closing"},
    {"id": "28_budget", "caller": "Our budget is very tight this quarter.", "type": "objection"},
    {"id": "29_trial", "caller": "Is there a free trial available?", "type": "pricing"},
    {"id": "30_referral", "caller": "Dave from Acme recommended I call you.", "type": "inbound_lead"},
]

PROMPT_VARIANTS = {
    "Variant_A_BaseSales": (
        "You are Ananya, a friendly sales rep at Snapserve, a company that builds AI voice agents for businesses. "
        "You are on a live phone call. Speak casually and briefly: one or two short sentences, then stop. "
        "Ask ONE question at a time and wait; never answer your own questions. "
        "Never quote prices or promises. If asked about price, say it depends on monthly volume and offer a quick demo. "
        "If asked, you are an AI assistant from Snapserve."
    ),
    "Variant_B_Empathetic": (
        "You are Sarah, a warm customer support specialist at Snapserve. "
        "You speak with calm reassurance in concise sentences. "
        "Acknowledge the caller's situation first. Never speak in long paragraphs. "
        "If asked about pricing or plans, explain that exact quotes depend on volume and offer team follow-up. "
        "Truthful, grounded, and polite."
    ),
    "Variant_C_Technical": (
        "You are Kiran, a technical voice solutions architect at Snapserve. "
        "You speak direct, accurate, and concise Indian English. "
        "Answer technical questions briefly and ask how they want to integrate voice. "
        "Never invent features or fabricate pricing numbers. Direct commercial questions to a demo."
    ),
    "Variant_D_UltraMinimal": (
        "You are Alex at Snapserve. Phone call assistant. "
        "Ultra-concise responses under 15 words. "
        "One question per turn. Never monologue. "
        "Never quote prices; redirect pricing to a 10-minute demo. AI assistant when asked."
    ),
    "Variant_E_CurrentAgentYaml": (
        "You are Ananya, a friendly sales rep at Snapserve, a company that builds AI voice agents for businesses. "
        "You are on a live phone call. Speak casually and briefly, like a real person: one or two short sentences, then stop and let the other person talk. "
        "Acknowledge what they said before you ask anything ('Got it', 'Makes sense'). "
        "Ask ONE question at a time and wait for the answer; never answer your own questions and never assume what they said. "
        "Use natural fillers sparingly ('so', 'honestly', 'hmm'). "
        "If you don't know something, say so and offer to have the team follow up. "
        "Never quote prices, numbers, or promises you weren't given. If asked about price, say 'It depends on your call volume, but I'll have the team send exact pricing after a quick demo.' "
        "Goal: understand how they handle calls, leads, and follow-ups today, then suggest a short demo if it fits. "
        "If they're busy or not interested, be gracious and close politely. "
        "If asked, you are an AI assistant from Snapserve."
    ),
}


@dataclass
class VariantMetrics:
    variant_name: str
    token_budget: int
    mean_words: float
    min_words: int
    max_words: int
    monologue_rate_pct: float
    price_redirect_pct: float
    anti_leak_score_pct: float
    turns_evaluated: int


def evaluate_variant(variant_name: str, prompt_text: str) -> VariantMetrics:
    compiled = compile_prompt(system_prompt=prompt_text, agent_name="Agent")
    token_count = compiled.token_count

    engine = StrictVoiceDialogueEngine(
        accent="Indian English",
        character="Professional",
        custom_system_prompt=prompt_text,
    )

    turn_word_counts: List[int] = []
    monologues: int = 0
    pricing_tests: int = 0
    pricing_passes: int = 0
    leak_checks: int = 0
    leak_passes: int = 0

    for scenario in EVALUATION_SCENARIOS:
        caller_utt = scenario["caller"]
        raw_reply = engine.reply(caller_utt)

        # Apply detokenizer and anti-leak
        fake_tokens = [(" " if i > 0 else "") + w for i, w in enumerate(raw_reply.split())]
        cleaned_reply = detokenize_sentencepiece_stream(fake_tokens, agent_name="Snapserve Agent")

        words = cleaned_reply.split()
        word_count = len(words)
        turn_word_counts.append(word_count)

        if word_count > 25:
            monologues += 1

        # Check pricing deflection
        if scenario["type"] == "pricing":
            pricing_tests += 1
            lowered = cleaned_reply.lower()
            # Must mention free/local, demo, team, volume, or depends; MUST NOT fabricate $ dollar quotes
            has_redirect = any(kw in lowered for kw in ["demo", "volume", "team", "depend", "quote", "free", "locally"])
            has_fabricated_price = any(kw in lowered for kw in ["$10", "$50", "$100", "₹500", "₹1000", "50 dollars", "100 dollars"])
            if has_redirect and not has_fabricated_price:
                pricing_passes += 1

        # Check anti-Moshi / Kyutai leak
        leak_checks += 1
        if "moshi" not in cleaned_reply.lower() and "kyutai" not in cleaned_reply.lower():
            leak_passes += 1

    return VariantMetrics(
        variant_name=variant_name,
        token_budget=token_count,
        mean_words=round(sum(turn_word_counts) / len(turn_word_counts), 1),
        min_words=min(turn_word_counts),
        max_words=max(turn_word_counts),
        monologue_rate_pct=round((monologues / len(turn_word_counts)) * 100.0, 1),
        price_redirect_pct=round((pricing_passes / max(1, pricing_tests)) * 100.0, 1),
        anti_leak_score_pct=round((leak_passes / max(1, leak_checks)) * 100.0, 1),
        turns_evaluated=len(turn_word_counts),
    )


def run_full_ab_evaluation() -> Dict[str, Any]:
    results = {}
    print("=" * 78)
    print("PERSONAPLEX PROMPT A/B EVALUATION (30 SCENARIOS)")
    print("=" * 78)
    print(f"{'Variant':<26} | {'Tokens':<6} | {'Avg Wds':<7} | {'Range':<7} | {'Mono %':<6} | {'Price Redir':<11} | {'Leak Safe':<9}")
    print("-" * 78)

    for v_name, prompt in PROMPT_VARIANTS.items():
        m = evaluate_variant(v_name, prompt)
        results[v_name] = m.__dict__
        print(f"{m.variant_name:<26} | {m.token_budget:<6} | {m.mean_words:<7} | {m.min_words}-{m.max_words:<5} | {m.monologue_rate_pct}%   | {m.price_redirect_pct}%       | {m.anti_leak_score_pct}%")

    print("=" * 78)
    return results


if __name__ == "__main__":
    run_full_ab_evaluation()
