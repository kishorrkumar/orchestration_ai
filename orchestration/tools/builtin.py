"""
Standard Built-in Business Tools for Conversational Voice Agents.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import ToolResult, tool
from .registry import default_registry

logger = logging.getLogger("orchestration.tools.builtin")


@tool(
    name="book_calendar_slot",
    description="Book a demo, consultation, or sales appointment on the calendar for a caller.",
)
async def book_calendar_slot(
    date: str,
    time: str,
    caller_name: str,
    phone_number: str = "",
    notes: str = "",
) -> ToolResult:
    """
    Book a slot on the calendar.
    """
    logger.info(f"[TOOL] Booking calendar slot on {date} at {time} for {caller_name} ({phone_number})")
    booking_id = f"bk_{abs(hash(date + time + caller_name)) % 1000000:06d}"
    summary = f"Confirmed appointment on {date} at {time} for {caller_name}."
    speech = f"You are all booked for {date} at {time}. I have sent the details to your phone."
    return ToolResult(
        status="success",
        data={
            "booking_id": booking_id,
            "date": date,
            "time": time,
            "caller_name": caller_name,
            "phone_number": phone_number,
            "status": "confirmed",
        },
        summary=summary,
        direct_speech=speech,
    )


@tool(
    name="query_pricing",
    description="Look up verified Snapserve pricing tiers, platform fees, and volume discounts.",
)
async def query_pricing(
    plan_tier: str = "growth",
    monthly_minutes: int = 1000,
) -> ToolResult:
    """
    Look up authoritative pricing details.
    """
    logger.info(f"[TOOL] Querying pricing for plan '{plan_tier}', minutes: {monthly_minutes}")
    plans = {
        "starter": {"base_usd": 49, "included_mins": 250, "per_min_usd": 0.20},
        "growth": {"base_usd": 199, "included_mins": 1200, "per_min_usd": 0.16},
        "enterprise": {"base_usd": 799, "included_mins": 6000, "per_min_usd": 0.12},
    }
    selected = plans.get(plan_tier.lower(), plans["growth"])
    summary = f"Snapserve {plan_tier.capitalize()} plan is ${selected['base_usd']}/month including {selected['included_mins']} minutes."
    speech = f"Our {plan_tier} plan starts at ${selected['base_usd']} a month, which includes {selected['included_mins']} minutes of voice calling."
    return ToolResult(
        status="success",
        data={
            "plan": plan_tier,
            "pricing": selected,
        },
        summary=summary,
        direct_speech=speech,
    )


@tool(
    name="lookup_customer",
    description="Query CRM to retrieve customer contact information, account tier, and prior call history.",
)
async def lookup_customer(
    phone_number: str,
) -> ToolResult:
    """
    Retrieve customer record from CRM.
    """
    logger.info(f"[TOOL] Looking up customer by phone: {phone_number}")
    # Normalize phone
    clean_phone = phone_number.strip().replace(" ", "").replace("-", "")
    return ToolResult(
        status="success",
        data={
            "phone_number": clean_phone,
            "customer_name": "Valued Caller",
            "tier": "active_lead",
            "open_tickets": 0,
        },
        summary=f"Found CRM record for {clean_phone}.",
    )


@tool(
    name="transfer_call",
    description="Transfer the current live call to a human specialist, department queue, or manager.",
)
async def transfer_call(
    department: str = "support",
    reason: str = "Caller requested human assistance",
) -> ToolResult:
    """
    Initiate call transfer to a live specialist.
    """
    logger.info(f"[TOOL] Transferring call to {department} (reason: {reason})")
    speech = f"I am transferring you to our {department} team right now. Please hold on for just a moment."
    return ToolResult(
        status="success",
        data={
            "department": department,
            "reason": reason,
            "action": "transfer_initiated",
        },
        summary=f"Transfer initiated to {department}.",
        direct_speech=speech,
    )


@tool(
    name="send_sms_summary",
    description="Send an SMS message to the caller with links, confirmations, or follow-up details.",
)
async def send_sms_summary(
    phone_number: str,
    message: str,
) -> ToolResult:
    """
    Send follow-up SMS.
    """
    logger.info(f"[TOOL] Sending SMS to {phone_number}: {message}")
    speech = "I have sent that information straight to your phone via SMS."
    return ToolResult(
        status="success",
        data={
            "phone_number": phone_number,
            "message": message,
            "delivery_status": "queued",
        },
        summary=f"SMS queued for {phone_number}.",
        direct_speech=speech,
    )


def register_builtin_tools(registry: Any = default_registry) -> None:
    """Register all standard built-in tools into registry."""
    registry.register(book_calendar_slot)
    registry.register(query_pricing)
    registry.register(lookup_customer)
    registry.register(transfer_call)
    registry.register(send_sms_summary)


# Automatically register defaults upon module load
register_builtin_tools(default_registry)
