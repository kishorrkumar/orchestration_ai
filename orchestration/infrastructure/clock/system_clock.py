"""System clock and deterministic test clock implementations of Clock protocol."""

from __future__ import annotations

from datetime import UTC, datetime

from orchestration.domain.protocols import Clock


class SystemClock(Clock):
    """Standard system clock providing current UTC time."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock(Clock):
    """Deterministic frozen clock for unit and integration testing."""

    def __init__(self, frozen_time: datetime | None = None) -> None:
        self._current = frozen_time or datetime(2026, 10, 5, 14, 30, 0, tzinfo=UTC)

    def now_utc(self) -> datetime:
        return self._current

    def advance(self, seconds: float) -> None:
        from datetime import timedelta
        self._current += timedelta(seconds=seconds)

    def set_time(self, new_time: datetime) -> None:
        self._current = new_time
