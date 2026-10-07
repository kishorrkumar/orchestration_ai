"""
Built-in Internal Tools for Engine B.
Provides natural programmatic call termination (end_call).
"""

from __future__ import annotations

from typing import Any


def create_end_call_tool_definition() -> dict[str, Any]:
    """Returns OpenAI/Anthropic compatible function schema for end_call."""
    return {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "Call this function when the conversation is finished and it is time to politely end the call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short explanation for ending the call (e.g., 'completed', 'user_goodbye')",
                    }
                },
                "required": [],
            },
        },
    }
