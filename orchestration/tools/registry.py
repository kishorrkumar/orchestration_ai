"""
Tool Registry & Execution Dispatcher.

Manages tool definitions, validates JSON arguments, and coordinates async execution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import ToolDefinition, ToolResult

logger = logging.getLogger("orchestration.tools.registry")


class ToolRegistry:
    """Central registry for structured JSON tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool_def.name] = tool_def
        logger.info(f"Registered tool: '{tool_def.name}'")

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Retrieve tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI/Anthropic format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        """
        Execute a tool by name with arguments (either parsed dict or raw JSON string).
        """
        tool_def = self._tools.get(name)
        if not tool_def:
            logger.warning(f"Attempted execution of unknown tool: '{name}'")
            return ToolResult(
                status="error",
                error_message=f"Tool '{name}' is not registered.",
                summary=f"Unknown tool '{name}'",
            )

        parsed_args: dict[str, Any] = {}
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as jde:
                return ToolResult(
                    status="error",
                    error_message=f"Invalid JSON arguments: {jde}",
                    summary="Failed to parse arguments JSON",
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments

        logger.info(f"Executing tool '{name}' with args: {parsed_args}")
        try:
            result = await tool_def.handler(**parsed_args)
            return result
        except Exception as err:
            logger.error(f"Error executing tool '{name}': {err}", exc_info=True)
            return ToolResult(
                status="error",
                error_message=str(err),
                summary=f"Tool execution failed: {err}",
            )


# Global default registry instance
default_registry = ToolRegistry()
