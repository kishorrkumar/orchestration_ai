"""
Structured JSON Tool Calling & Schema Definition Framework.

Provides standard JSON Schema extraction, argument validation,
and asynchronous function dispatching for conversational voice agents.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, get_type_hints


@dataclass
class ToolResult:
    """Standardized result returned by an executed tool."""
    status: str  # "success" | "error"
    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    direct_speech: str | None = None  # Optional deterministic spoken response
    error_message: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "data": self.data,
            "summary": self.summary,
            "direct_speech": self.direct_speech,
            "error_message": self.error_message,
        }


@dataclass
class ToolDefinition:
    """Structured JSON schema description conforming to OpenAI/Anthropic function calling specs."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, ToolResult]]

    def to_schema(self) -> dict[str, Any]:
        """Generate OpenAI/Anthropic compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _python_type_to_json_type(py_type: Any) -> str:
    """Map standard Python types to JSON Schema types."""
    if py_type in (str,):
        return "string"
    elif py_type in (int,):
        return "integer"
    elif py_type in (float,):
        return "number"
    elif py_type in (bool,):
        return "boolean"
    elif py_type in (list, set, tuple):
        return "array"
    elif py_type in (dict,):
        return "object"
    return "string"


def tool(name: str | None = None, description: str | None = None) -> Callable:
    """
    Decorator to convert an async Python function into a structured JSON tool.
    Extracts parameter types, defaults, and docstrings automatically.
    """
    def decorator(fn: Callable[..., Coroutine[Any, Any, Any]]) -> ToolDefinition:
        func_name = name or fn.__name__
        doc = description or inspect.getdoc(fn) or f"Execute {func_name}"

        sig = inspect.signature(fn)
        type_hints = get_type_hints(fn)

        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = type_hints.get(param_name, str)
            json_type = _python_type_to_json_type(param_type)

            prop_def: dict[str, Any] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                prop_def["default"] = param.default

            properties[param_name] = prop_def

        parameters_schema = {
            "type": "object",
            "properties": properties,
            "required": required_params,
        }

        async def wrapped_handler(*args: Any, **kwargs: Any) -> ToolResult:
            try:
                res = await fn(*args, **kwargs)
                if isinstance(res, ToolResult):
                    return res
                elif isinstance(res, dict):
                    return ToolResult(status="success", data=res, summary=str(res))
                else:
                    return ToolResult(status="success", data={"result": res}, summary=str(res))
            except Exception as err:
                return ToolResult(
                    status="error",
                    error_message=str(err),
                    summary=f"Tool {func_name} failed: {err}",
                )

        return ToolDefinition(
            name=func_name,
            description=doc.strip(),
            parameters=parameters_schema,
            handler=wrapped_handler,
        )

    return decorator
