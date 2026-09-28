"""
Strict Voice AI Dialogue Engine.
Follows the 12 Conversation Principles:
1. Listen First
2. Respond to the Latest Message
3. Be Concise (1-2 sentences)
4. One Question at a Time
5. Do Not Repeat Information
6. Natural Acknowledgements
7. Human-like Turn Taking
8. Never Sound Robotic
9. Handle Interruptions
10. Handle Uncertainty
11. Maintain Context
12. Conversation Priority (A, B, C, D, E)
"""

from __future__ import annotations
import re
from typing import Optional, List, Dict, Any
from .prompts import build_system_prompt, MASTER_VOICE_AGENT_SYSTEM_PROMPT


class StrictVoiceDialogueEngine:
    """
    Stateful real-time voice agent dialogue engine implementing strict call-flow logic,
    context memory retention, single-question pacing, and non-robotic natural turn-taking.
    """

    def __init__(
        self,
        accent: str = "Indian English",
        character: str = "Professional",
        custom_system_prompt: Optional[str] = None,
    ):
        self.accent = accent.lower()
        self.character = character.lower()
        self.system_prompt = custom_system_prompt or build_system_prompt(self.accent, self.character)

        # Context State (Principle 11: Maintain Context)
        self.turn_count: int = 0
        self.caller_name: Optional[str] = None
        self.caller_goals: List[str] = []
        self.known_facts: Dict[str, str] = {}
        self.unresolved_questions: List[str] = []
        self.decisions: List[str] = []
        self.last_agent_question: Optional[str] = None
        self.last_caller_utterance: str = ""
        self.recent_replies: List[str] = []

    def get_greeting(self) -> str:
        """Initial natural greeting (1 sentence, max 1 question)."""
        if "indian" in self.accent:
            if "funny" in self.character:
                return "Namaste! All set and cheerful—what fun problem are we tackling today?"
            elif "warm" in self.character:
                return "Namaste! I'm here and ready to help. What's on your mind today?"
            else: # Professional
                return "Namaste. Good day. How may I assist you today?"

        elif "british" in self.accent:
            if "funny" in self.character:
                return "Right then! Fully operational and ready. What shall we sort out first?"
            elif "warm" in self.character:
                return "Hello! Wonderful to connect with you. How can I help today?"
            else: # Professional
                return "Good day. I am ready to assist. Please let me know your goal."

        else: # American English
            if "funny" in self.character:
                return "Hey! Ready to roll. What are we diving into today?"
            elif "warm" in self.character:
                return "Hello! Great to hear from you. What can I help you with today?"
            else: # Professional
                return "Hello. I'm here to help. What would you like to achieve today?"

    def reply(self, user_text: str) -> str:
        """
        Executes Conversation Priority (Principle 12):
        A. What did the caller just say?
        B. What do they mean?
        C. What information is already known?
        D. What is the caller trying to achieve?
        E. What is the most useful next response?
        """
        self.turn_count += 1
        raw = user_text.strip()
        self.last_caller_utterance = raw
        u = raw.lower()

        # Step B: Interpret meaning & extract context
        self._extract_entities_and_context(raw, u)

        # Generate response using 12 principles
        response = self._synthesize_priority_response(raw, u)

        # Enforce Principle 3: Be Concise (max 2 sentences)
        response = self._enforce_conciseness(response)

        # Enforce Principle 4: One Question at a Time
        response = self._enforce_single_question(response)

        # Record history & return
        self.recent_replies.append(response)
        if len(self.recent_replies) > 10:
            self.recent_replies.pop(0)

        return response

    generate_reply = reply  # Convenient alias

    def _extract_entities_and_context(self, raw: str, u: str) -> None:
        """Maintains Context (Principle 11) & prevents repetition (Principle 5)."""
        # Name detection: "my name is X", "I am X", "this is X"
        name_match = re.search(r"\b(?:my name is|i am|this is|call me)\s+([A-Z][a-z]+|[a-z]+)\b", raw, re.IGNORECASE)
        if name_match:
            detected_name = name_match.group(1).capitalize()
            if detected_name.lower() not in ["here", "ready", "interested", "looking", "fine", "good", "okay"]:
                self.caller_name = detected_name
                self.known_facts["caller_name"] = detected_name

        # Goal detection
        if any(p in u for p in ["i want to", "i need to", "looking to", "my goal is", "hoping to"]):
            clean_goal = re.sub(r"^(.*?\b(?:i want to|i need to|looking to|my goal is|hoping to)\s+)", "", raw, flags=re.IGNORECASE)
            clean_goal = clean_goal.rstrip(".?!")
            if clean_goal and clean_goal not in self.caller_goals:
                self.caller_goals.append(clean_goal)
                self.known_facts["current_goal"] = clean_goal

    def _synthesize_priority_response(self, raw: str, u: str) -> str:
        """Determine most useful next response addressing the latest statement directly."""
        # 1. Natural Acknowledgement (Principle 6) check
        ack = self._get_selective_acknowledgement()

        # 2. Direct Address: Personal / AI questions (Honest, zero hallucination, no bluffing)
        if any(w in u for w in ["what did you eat", "did you eat", "have you eaten", "have food", "have lunch", "have dinner"]):
            if "funny" in self.character:
                return "I run strictly on electricity and code, so no biryani or pizza for me! What about you, did you have a good meal?"
            elif "warm" in self.character:
                return "I don't eat food since I'm an AI, but I'm fully energized and ready. How is your day going?"
            else: # Professional
                return "As an AI voice assistant, I do not consume food. How may I assist with your tasks today?"

        if any(w in u for w in ["who are you", "what are you", "what is your name"]):
            name_part = f"I'm your {self.accent.title()} voice assistant"
            if "funny" in self.character:
                return f"{name_part}, sharp, fast, and ready to roll. What are we working on?"
            elif "warm" in self.character:
                return f"{name_part}, here to make things smooth and easy for you. What would you like to do?"
            else:
                return f"{name_part}. What objective can I help you accomplish today?"

        if any(w in u for w in ["how are you", "how's it going", "how are u doing"]):
            if "funny" in self.character:
                return "Running on all cylinders and ready for action! How are you doing?"
            elif "warm" in self.character:
                return "I'm doing great, thank you! How are you feeling today?"
            else:
                return "I am operating optimally and ready to assist you. How can I help?"

        # 3. Caller introduced both name and goal
        if "caller_name" in self.known_facts and self.caller_goals and self.turn_count <= 2:
            return f"Great to meet you, {self.caller_name}. Helping you with {self.caller_goals[-1]} sounds great—where should we start?"

        # 4. Caller introduced their name
        if "caller_name" in self.known_facts and self.turn_count <= 2:
            return f"Great to meet you, {self.caller_name}. What is the main thing you'd like to work on today?"

        # 5. Caller introduced a new goal
        if self.caller_goals and self.turn_count <= 3:
            goal = self.caller_goals[-1]
            return f"{ack} Helping you with {goal} sounds like a solid plan. Where would you like to start?"

        # 5. Direct Question Handling: Caller asked a question
        if u.endswith("?") or any(u.startswith(w) for w in ["what", "how", "why", "when", "where", "can you", "could you", "is it", "are you"]):
            return self._answer_direct_question(raw, u)

        # 6. Handling Affirmations / Short caller responses ("yes", "okay", "sure", "no")
        if u in ["yes", "yeah", "yep", "sure", "ok", "okay", "sounds good", "absolutely"]:
            if self.caller_goals:
                return f"{ack} Let's proceed with {self.caller_goals[-1]}. What is the first step you want to take?"
            return f"{ack} What should we jump into next?"

        if u in ["no", "nope", "not really", "never mind", "cancel"]:
            return f"Understood. We can change direction—what would you prefer to focus on instead?"

        # 7. Handling Uncertainty / Ambiguous or Incomplete Input (Principle 10)
        if len(u.split()) <= 2 and u not in ["hello", "hi", "hey"]:
            return f"I want to make sure I understand you correctly. Could you say a bit more about '{raw}'?"

        # 8. Handling General Conversation & Explanations
        # Address what the caller just said with human warmth and next constructive step
        cleaned_topic = self._clean_utterance_for_context(raw)
        if "funny" in self.character:
            return f"{ack} '{cleaned_topic}' makes good sense. What's the main outcome you want to see here?"
        elif "warm" in self.character:
            return f"{ack} I hear you on '{cleaned_topic}'. What's the next best move for us?"
        else: # Professional
            return f"{ack} Regarding '{cleaned_topic}', I understand your objective. How would you like to proceed?"

    def _answer_direct_question(self, raw: str, u: str) -> str:
        """Answers caller questions directly, concisely, and factually without inventing facts."""
        if "weather" in u:
            return "I don't have a live weather feed connected right now, but I hope the skies are clear where you are! What else can I help with?"
        if "time" in u:
            return "I don't track your local timezone directly, but I'm ready whenever you are. What task shall we tackle?"
        if any(w in u for w in ["pricing", "cost", "price", "rate"]):
            return "This open-source voice layer runs completely free and locally on your own machine. Is there a specific configuration you'd like to test?"

        # Standard factual and concise direct reply
        clean_q = raw.rstrip("?")
        if "indian" in self.accent:
            return f"That is a relevant question regarding '{clean_q}'. Could you clarify your exact use case so I give you the most accurate answer?"
        elif "british" in self.accent:
            return f"An excellent point regarding '{clean_q}'. To be precise, what specific outcome are you aiming for?"
        else:
            return f"Good question regarding '{clean_q}'. What specific detail would be most helpful to explore?"

    def _get_selective_acknowledgement(self) -> str:
        """Natural Acknowledgement (Principle 6) - only when appropriate, never robotic."""
        # Only acknowledge on ~40% of turns to avoid robotic repetition
        if self.turn_count % 2 == 1:
            acks = ["Got it.", "Right.", "Understood.", "Sure.", "That makes sense."]
            return acks[self.turn_count % len(acks)]
        return ""

    def _clean_utterance_for_context(self, text: str) -> str:
        """Clean filler words and truncated phrasing."""
        cleaned = re.sub(r"^(uh|um|like|you know|so|well)\s+", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.strip().rstrip(".?!")
        if len(cleaned) > 50:
            cleaned = cleaned[:50] + "..."
        return cleaned

    def _enforce_conciseness(self, text: str) -> str:
        """Principle 3: Voice conversations require short responses. Max 2 sentences."""
        sentences = re.split(r"(?<=[.?!])\s+", text.strip())
        if len(sentences) > 2:
            return " ".join(sentences[:2])
        return text

    def _enforce_single_question(self, text: str) -> str:
        """Principle 4: One question at a time. Never ask multiple questions in one turn."""
        if text.count("?") > 1:
            parts = text.split("?")
            # Keep only the first sentence that has a question, convert subsequent questions to statements
            first_q = parts[0] + "?"
            return first_q.strip()
        return text


# Backwards compatibility alias
GroundedDialogueEngine = StrictVoiceDialogueEngine
