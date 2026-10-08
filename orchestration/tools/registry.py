"""
Asynchronous Tool & Business Logic Registry for Voice Orchestration.

Architecture Note on PersonaPlex Speech-to-Speech:
NVIDIA PersonaPlex-7B-v1 is an end-to-end neural speech-to-speech foundation model (Mimi + Moshi LM).
Unlike discrete cascaded pipelines (STT -> LLM -> TTS), PersonaPlex does not support native function calling / tool calling
tokens mid-speech.
This ToolRegistry defines the orchestrator extension interface for external business logic and retrieval.
In the current native S2S engine, tool outputs (such as time, user account status, or RAG context) are pre-fetched
and injected into prompt conditioning at connection time. In the later modular cascaded engine (Engine B), tools
can be called dynamically by the LLM during conversation turns.
"""

from __future__ import annotations

import datetime
import inspect
import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger("orchestration.tools")


class Tool:
    """Represents a callable tool with metadata and JSON schema."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        parameters_schema: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters_schema = parameters_schema or {}

    async def execute(self, **kwargs: Any) -> Any:
        try:
            if inspect.iscoroutinefunction(self.handler):
                return await self.handler(**kwargs)
            return self.handler(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{self.name}': {e}", exc_info=True)
            return {"error": str(e)}


class ToolRegistry:
    """Registry managing external tool definitions and execution."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in registry")
        return await tool.execute(**kwargs)


# ------------------------------------------------------------------------------
# Example Tool: get_time
# ------------------------------------------------------------------------------
async def get_time_tool(timezone: str = "Asia/Kolkata") -> dict[str, str]:
    """Returns formatted local time, weekday, and date for a given timezone."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("UTC")

    now = datetime.datetime.now(tz)
    return {
        "timezone": timezone,
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "weekday": now.strftime("%A"),
        "date": now.strftime("%B %d, %Y"),
        "iso": now.isoformat(),
    }


default_tool_registry = ToolRegistry()
default_tool_registry.register(
    Tool(
        name="get_time",
        description="Get the current local time and date for the caller's timezone.",
        handler=get_time_tool,
        parameters_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "default": "Asia/Kolkata"},
            },
        },
    )
)
