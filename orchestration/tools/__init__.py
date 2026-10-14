"""
Structured JSON Tools & Function Calling Package.
"""

from .base import ToolDefinition, ToolResult, tool
from .builtin import (
    book_calendar_slot,
    lookup_customer,
    query_pricing,
    register_builtin_tools,
    send_sms_summary,
    transfer_call,
)
from .registry import ToolRegistry, default_registry

__all__ = [
    "tool",
    "ToolDefinition",
    "ToolResult",
    "ToolRegistry",
    "default_registry",
    "book_calendar_slot",
    "query_pricing",
    "lookup_customer",
    "transfer_call",
    "send_sms_summary",
    "register_builtin_tools",
]
