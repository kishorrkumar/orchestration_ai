"""Type-safe prefixed ID generators for entities and error correlation."""

import uuid


def _gen_id(prefix: str) -> str:
    """Generate a clean, collision-resistant prefixed identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def new_error_id() -> str:
    return _gen_id("err")


def new_agent_id() -> str:
    return _gen_id("agt")


def new_session_id() -> str:
    return _gen_id("ses")


def new_turn_id() -> str:
    return _gen_id("trn")
