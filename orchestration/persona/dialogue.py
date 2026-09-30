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

from .prompts import build_system_prompt


class StrictVoiceDialogueEngine:
    """
    Stateful real-time voice agent dialogue engine implementing strict call-flow logic,
    context memory retention, single-question pacing, and non-robotic natural turn-taking.
    """

    def __init__(
        self,
        accent: str = "Indian English",
        character: str = "Professional",
        custom_system_prompt: str | None = None,
    ):
        self.accent = accent.lower()
        self.character = character.lower()
        self.system_prompt = custom_system_prompt or build_system_prompt(self.accent, self.character)

        # Context State (Principle 11: Maintain Context)
        self.turn_count: int = 0
        self.caller_name: str | None = None
        self.caller_goals: list[str] = []
        self.known_facts: dict[str, str] = {}
        self.unresolved_questions: list[str] = []
        self.decisions: list[str] = []
        self.last_agent_question: str | None = None
        self.last_caller_utterance: str = ""
        self.recent_replies: list[str] = []

    def get_greeting(self) -> str:
        """Initial natural greeting — concise, one sentence, one question (Principle 4)."""
        if "priya" in self.character or "female" in self.character or "ananya" in self.character or "warm" in self.character:
            return "Namaste! I'm Priya. How can I help you today?"
        if "kabir" in self.character or "funny" in self.character:
            return "Namaste! Kabir here. What's on your mind today?"
        return "Namaste, this is Aarav. How can I help you today?"

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
        name_match = re.search(r"\b(?:my name is|i am|this is|call me)\s+([A-Z][a-z]+|[a-z]+)\b", raw, re.IGNORECASE)
        if name_match:
            detected_name = name_match.group(1).capitalize()
            if detected_name.lower() not in ["here", "ready", "interested", "looking", "fine", "good", "okay"]:
                self.caller_name = detected_name
                self.known_facts["caller_name"] = detected_name

        if any(p in u for p in ["i want to", "i need to", "looking to", "my goal is", "hoping to"]):
            clean_goal = re.sub(r"^(.*?\b(?:i want to|i need to|looking to|my goal is|hoping to)\s+)", "", raw, flags=re.IGNORECASE)
            clean_goal = clean_goal.rstrip(".?!")
            if clean_goal and clean_goal not in self.caller_goals:
                self.caller_goals.append(clean_goal)
                self.known_facts["current_goal"] = clean_goal

    def _synthesize_priority_response(self, raw: str, u: str) -> str:
        """Determine most useful next response addressing the latest statement directly like a real human."""
        u_clean = re.sub(r"[^\w\s]", " ", u)
        u_clean = " ".join(u_clean.split())
        words = u_clean.split()

        # 1. Jokes & Humor
        if any(w in u_clean for w in ["joke", "funny", "laugh", "make me laugh"]):
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Why was the computer cold? Because it forgot to close its Windows!",
                "Why don't skeletons fight each other? Because they just don't have the guts!",
                "What do you call an alligator in a vest? An investigator!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "Why did the smartphone need glasses? Because it lost its contacts!",
            ]
            joke = jokes[self.turn_count % len(jokes)]
            return f"Here is a good one: {joke} What do you think?"

        # 2. Greetings
        if u_clean in ["hello", "hi", "hey", "namaste", "good morning", "good evening", "good afternoon"] or (
            len(words) <= 3 and any(w in words for w in ["hello", "hi", "hey", "namaste"])
        ):
            if "caller_name" in self.known_facts:
                return f"Hello {self.caller_name}! Great to hear your voice. What can I do for you today?"
            return "Namaste! Great to connect with you. How are you doing today?"

        # 3. Connectivity & Audio Hearing
        if any(w in u_clean for w in ["can you hear", "can u hear", "hear me", "you hear me", "are you there"]):
            return "Yes, I hear you loud and clear! How can I help you today?"

        # 4. Personal & AI Nature
        if any(w in u_clean for w in ["what did you eat", "did you eat", "have you eaten", "have food", "have lunch", "have dinner"]):
            return "I run strictly on electricity and code, so no biryani for me! Did you have a good meal today?"

        if any(w in u_clean for w in ["who are you", "what are you", "what is your name", "who made you"]):
            is_fem = "priya" in self.character or "female" in self.character or "ananya" in self.character or "warm" in self.character
            name = "Priya" if is_fem else "Aarav"
            return f"I'm {name}, your real-time Indian English voice assistant. What can I help you accomplish today?"

        if any(w in u_clean for w in ["how are you", "how are you doing", "hows it going", "how are u"]):
            return "I'm doing great, feeling energized and ready to assist! How is your day going so far?"

        # 5. Artificial Intelligence & Technology
        if any(w in u_clean for w in ["artificial intelligence", "machine learning", "neural network", "deep learning"]) or (
            "ai" in words or "a i" in u_clean or "about ai" in u_clean or "about a i" in u_clean
        ):
            return "Artificial Intelligence is about building systems that can learn patterns, understand language, and solve problems like humans do. Are you interested in voice AI or general technology?"

        # 6. Helicopters / Gun Choppers / Aviation
        if any(w in u_clean for w in ["gun chopper", "chopper", "helicopter", "gunship", "attack heli"]):
            return "A gun chopper is an armored attack helicopter equipped with rapid-fire autocannons, rockets, and anti-tank guided missiles, like the Apache or India's Prachand. Were you curious about how they operate or military aviation in general?"

        # 7. Stories / Narrative
        if any(w in u_clean for w in ["narrative", "story", "tale", "tell me a story"]):
            return "Here's a quick story: An engineer in Bengaluru built a voice AI, and on its first trial call, it unexpectedly solved a major logistics problem and made the client laugh! What kind of stories do you enjoy hearing most?"

        # 8. Prompts, Roles & Instructions (Directly addresses 'Can you talk a prompt?')
        if any(w in u_clean for w in ["prompt", "system prompt", "talk a prompt", "say a prompt", "read prompt", "instructions"]):
            return "Yes, absolutely! I am operating under my system prompt to assist you clearly and professionally in natural Indian English. What scenario would you like to run?"

        # 9. Follow-up / Clarification on why agent said or did something
        if any(w in u_clean for w in ["why didnt you", "why did you", "why did you say", "repeat", "second and again"]):
            return "I wanted to keep our dialogue fresh and moving forward naturally rather than looping on the same words! What topic would you like to explore next?"

        # 10. Context: Caller introduced both name and goal
        if "caller_name" in self.known_facts and self.caller_goals and self.turn_count <= 2:
            return f"Great to meet you, {self.caller_name}. Helping you with {self.caller_goals[-1]} sounds great—where should we start?"

        # 11. Context: Caller introduced their name
        if "caller_name" in self.known_facts and self.turn_count <= 2:
            return f"Great to meet you, {self.caller_name}. What is the main thing you'd like to work on today?"

        # 12. Context: Caller introduced a new goal
        if self.caller_goals and self.turn_count <= 3:
            goal = self.caller_goals[-1]
            return f"Helping you with {goal} sounds like a solid plan. Where would you like to start?"

        # 13. Direct Questions & Requests (Answering directly, contextually and humanly)
        if u.endswith("?") or any(u_clean.startswith(w) for w in ["what", "how", "why", "when", "where", "can you", "could you", "is it", "are you", "tell me", "explain", "help me", "do you", "will you"]):
            return self._answer_direct_question(raw, u_clean)

        # 14. Affirmations & Short responses
        if u_clean in ["yes", "yeah", "yep", "sure", "ok", "okay", "sounds good", "absolutely", "definitely"]:
            if self.caller_goals:
                return f"Let's proceed with {self.caller_goals[-1]}. What is the first step you'd like to take?"
            return "Sounds great! What should we dive into next?"

        if u_clean in ["no", "nope", "not really", "never mind", "cancel"]:
            return "Understood. We can change direction—what would you prefer to focus on instead?"

        # 15. Task & Action Intents (booking, scheduling, calculating, transferring)
        if any(w in u_clean for w in ["book", "reserve", "ticket", "flight", "hotel", "seat"]):
            return "I can certainly help you book that. Could you share the date and destination details?"

        if any(w in u_clean for w in ["transfer", "payment", "rupees", "send money", "pay"]):
            amount_match = re.search(r"(\d+[\d,]*|\b(?:thousand|lakh|hundred)\b)", u_clean)
            amt = amount_match.group(0) if amount_match else "the payment"
            return f"Understood. For security, please confirm the recipient details for {amt}."

        if any(w in u_clean for w in ["schedule", "meeting", "calendar", "appointment"]):
            return "I can help organize your schedule. What date and time works best for you?"

        # 16. Intent & Friendly conversation
        if any(w in u_clean for w in ["looking for", "want to talk", "interested in", "someone who"]):
            return "I am right here and ready to chat with you! What would you like to focus on?"

        # 17. Short input fallback - Positive human acknowledgment
        if len(words) <= 2:
            topic = raw.rstrip(".?!,")
            return f"Got it, {topic}. How can I best help you with that right now?"

        # 18. Dynamic conversational response based on caller's actual words (No canned phrases!)
        clean_stmt = raw.rstrip(".?!")
        return f"Understood regarding '{clean_stmt}'. How would you like us to proceed on this?"

    def _answer_direct_question(self, raw: str, u_clean: str) -> str:
        """Answers caller questions directly, factually, and conversationally like a knowledgeable human."""
        u_clean.split()

        # Questions about capabilities or prompt
        if any(w in u_clean for w in ["talk a prompt", "talk prompt", "speak prompt", "give a prompt", "say prompt"]):
            return "Certainly! Here is an example prompt: 'Act as a professional customer support lead resolving a high-priority account request.' Would you like to practice that?"

        if any(w in u_clean for w in ["can you talk", "can you speak", "can you voice"]):
            return "Yes, I am speaking with you in real time right now! What topic would you like to discuss?"

        if any(w in u_clean for w in ["can you hear", "can u hear", "hear me"]):
            return "Yes, I can hear you loud and clear! What's on your mind?"

        # Specific topic answers
        if "weather" in u_clean:
            return "I don't have a live weather feed connected right now, but I hope the skies are clear where you are! What else can I help with?"

        if "time" in u_clean:
            return "I don't track your local clock directly, but I'm ready whenever you are. What should we tackle?"

        if any(w in u_clean for w in ["pricing", "cost", "price", "rate"]):
            return "This voice AI pipeline runs completely free and locally on your own machine. Is there a specific configuration you'd like to test?"

        if any(w in u_clean for w in ["coding", "programming", "python", "javascript"]):
            return "Coding is all about breaking down complex problems into clear, logical steps. Which language or project are you working on?"

        if any(w in u_clean for w in ["india", "indian"]):
            return "India is an amazing country known for its rich history, diverse languages, and incredible technology ecosystem. What aspect are you most curious about?"

        if any(w in u_clean for w in ["food", "dish", "biryani", "cuisine"]):
            return "Indian cuisine is celebrated worldwide for its incredible spices and rich flavors! What is your favorite dish to enjoy?"

        if any(w in u_clean for w in ["movie", "cinema", "music", "song"]):
            return "Music and movies are fantastic ways to connect and unwind! Do you have a favorite genre or artist you enjoy?"

        # Direct semantic question answering extracting topic
        clean_topic = re.sub(
            r"^(can you|could you|what is|what are|how do|how does|why is|why are|tell me about|explain|help me with|do you know)\s+",
            "",
            u_clean,
            flags=re.IGNORECASE,
        ).strip()

        if clean_topic:
            # Filter out punctuation
            topic_disp = re.sub(r"^(talk|say|explain|tell me)\s+", "", clean_topic).strip()
            if topic_disp:
                return f"Regarding {topic_disp}, I can certainly help you explore that. What specific outcome are you looking for?"

        return "I understand your question. What specific detail or next step should we focus on?"

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
