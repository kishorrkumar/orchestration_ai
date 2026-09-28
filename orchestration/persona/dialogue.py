"""
Intelligent, Grounded Conversational Dialogue Engine.
Prevents hallucinations and bluffs.
Adapts responses across 3 Accents (Indian, American, British English)
and 3 Characters (Professional, Funny, Confident & Warm).
"""

from __future__ import annotations
import re
from typing import Optional, List, Set, Dict
from ..rag.engine import default_rag_engine


class GroundedDialogueEngine:
    """
    Dialogue engine providing clean, accurate, non-hallucinatory speech turns.
    Zero bluffs, honest about capabilities, and stylistically tailored
    to chosen accent and character traits.
    """

    def __init__(
        self,
        accent: str = "American English",
        character: str = "Confident, Warm & Concise",
        rag_engine=None,
    ):
        self.accent = accent.lower()
        self.character = character.lower()
        self.rag = rag_engine or default_rag_engine
        self.turn_count = 0
        self.recent_replies: List[str] = []

    def get_greeting(self) -> str:
        # Accent & Character matrix for initial greeting
        if "indian" in self.accent:
            if "funny" in self.character:
                return "Namaste! All charged up and ready. What interesting thing are we solving today?"
            elif "professional" in self.character:
                return "Namaste. Good day. I am ready to assist you with your tasks or review your documents."
            else: # Confident, warm & concise
                return "Namaste! I am here and ready to help. What's on your mind today?"

        elif "british" in self.accent:
            if "funny" in self.character:
                return "Right then! Fully operational and remarkably cheerful. What shall we tackle first?"
            elif "professional" in self.character:
                return "Good day. I am at your service. Please let me know how I may assist you today."
            else: # Confident, warm & concise
                return "Hello! I am ready to help. What would you like to discuss today?"

        else: # American English
            if "funny" in self.character:
                return "Hey there! Ready to roll. Fire away with your questions or docs!"
            elif "professional" in self.character:
                return "Hello. I am ready to assist you with your workflow and inquiries. How can I help?"
            else: # Confident, warm & concise
                return "Hello! I'm here and ready to assist. What can I help you with today?"

    def reply(self, user_text: str) -> str:
        self.turn_count += 1
        raw = user_text.strip()
        u = raw.lower()

        # 1. First, check Document Knowledge Base (RAG)
        rag_resp = self.rag.generate_grounded_response(raw)
        if rag_resp and any(w in u for w in ["document", "file", "pdf", "text", "notes", "policy", "say", "according", "what"]):
            # Document inquiry with actual groundings
            ans = self._stylize(rag_resp)
            self.recent_replies.append(ans)
            return ans

        # If user explicitly asked about uploaded docs but no match found:
        if any(w in u for w in ["in the document", "in the file", "in my pdf", "what does the doc say", "from the file"]):
            docs = default_rag_engine.list_documents()
            if not docs:
                return "You haven't uploaded any documents yet. Drop a PDF or text file in the knowledge zone on the left, and I'll read it immediately."
            else:
                return f"I checked your {len(docs)} uploaded document(s), but that specific information is not mentioned in them."

        # 2. Personal / Physical inquiries about the AI (Zero hallucinations / No bluffs!)
        if any(w in u for w in ["what did you eat", "did you eat", "have you eaten", "have lunch", "have breakfast", "have dinner", "food"]):
            if "funny" in self.character:
                return "I run on pure electricity and Python code! No biryani or burgers for me, but I hope you had something delicious."
            elif "professional" in self.character:
                return "As an AI voice assistant, I do not consume food. I am powered by code and ready to assist your work."
            else:
                return "I don't eat food since I'm an AI, but I'm fully charged and ready to assist you!"

        if any(w in u for w in ["who are you", "what are you", "your name"]):
            if "indian" in self.accent:
                return "I am your voice assistant powered by the PersonaPlex orchestration engine. How can I help you today?"
            elif "british" in self.accent:
                return "I am your PersonaPlex voice agent, designed for realtime, full-duplex conversations. What can I do for you?"
            else:
                return "I am an AI voice assistant running on the PersonaPlex orchestration architecture. How can I assist you?"

        if any(w in u for w in ["how are you", "how are you doing", "how's it going", "how are u"]):
            if "funny" in self.character:
                return "Running at zero errors and maximum enthusiasm! How are things on your side?"
            elif "professional" in self.character:
                return "I am operating efficiently and ready to assist. How can I help you with your tasks?"
            else:
                return "I'm doing great, thank you! Ready to help you with whatever you need."

        if any(w in u for w in ["can you repair", "fix file", "file repair", "repair this", "fix the file"]):
            return "I can inspect, search, and extract knowledge from files you upload to the RAG zone on the left. If a file has corrupted data, I can read the extractable text for you."

        if any(w in u for w in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]):
            if "funny" in self.character:
                return "Hey! Great to hear from you. What's on the agenda today?"
            elif "professional" in self.character:
                return "Good day. Please let me know how I can be of service."
            else:
                return "Hello! I'm listening. How can I assist you?"

        if any(w in u for w in ["thank you", "thanks", "appreciate"]):
            if "funny" in self.character:
                return "You're very welcome! Always happy to help."
            elif "professional" in self.character:
                return "You are welcome. Please let me know if you require any further assistance."
            else:
                return "You're very welcome! Glad I could help."

        # 3. Direct factual answering (avoid generic rambling templates!)
        # Check if user is asking a clear question
        if u.startswith("what is") or u.startswith("explain") or u.startswith("how does"):
            # Clean direct answer
            cleaned_topic = re.sub(r"^(what is|explain|how does)\s+", "", u).rstrip("?.")
            if "quantum" in u:
                return "Quantum mechanics describes nature at the atomic scale, where energy is quantized and particles exhibit both wave and particle properties."
            elif "python" in u:
                return "Python is a high-level, interpreted programming language known for readable syntax and rich libraries for AI, web, and automation."
            elif "api" in u:
                return "An API, or Application Programming Interface, defines the rules and protocols that allow different software programs to communicate with each other."
            elif "personaplex" in u or "moshi" in u:
                return "PersonaPlex is NVIDIA's full-duplex speech-to-speech model based on Moshi architecture, processing audio at 24 kilohertz with 80 millisecond frames."
            else:
                return f"Regarding {cleaned_topic}: let me know what specific detail or document you'd like me to focus on."

        # 4. Safe, grounded conversational reply (never repeats verbatim, no bluffing)
        if len(u) < 4:
            return "I'm listening. Please go ahead."

        resp = self._format_clean_reply(raw)
        
        # De-duplicate
        if resp in self.recent_replies[-2:]:
            resp = "Understood. Please let me know how you'd like to proceed, or upload a document for us to review."

        self.recent_replies.append(resp)
        if len(self.recent_replies) > 8:
            self.recent_replies.pop(0)

        return resp

    def _stylize(self, text: str) -> str:
        """Add subtle character/accent touch to factual answers."""
        if "funny" in self.character:
            return f"{text} Pretty neat, right?"
        elif "professional" in self.character:
            return f"As documented: {text}"
        else:
            return text

    def _format_clean_reply(self, raw_input: str) -> str:
        """Generates a succinct, honest, non-bluffing response."""
        cleaned = raw_input.strip().rstrip(".?")
        if "funny" in self.character:
            return f"Got it: '{cleaned}'. Tell me a bit more so I can give you the best answer, or drop in a file for us to analyze."
        elif "professional" in self.character:
            return f"Understood regarding '{cleaned}'. Please let me know if you would like me to analyze a specific document or answer further questions."
        else:
            return f"Understood regarding '{cleaned}'. What specific aspect would you like to explore?"

    def generate_reply(self, user_text: str) -> str:
        return self.reply(user_text)
