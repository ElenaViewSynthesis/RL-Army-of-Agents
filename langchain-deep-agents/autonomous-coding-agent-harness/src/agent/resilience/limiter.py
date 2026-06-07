"""Async token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import os
import time


class TokenBucket:
    """Simple async token bucket."""

    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self.rate_per_minute = max(rate_per_minute, 1)
        self.capacity = max(burst, 1)
        self.tokens = float(self.capacity)
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.updated_at = now
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * (self.rate_per_minute / 60.0),
        )

    async def acquire(self) -> None:
        """Wait until one token is available."""
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                missing = 1 - self.tokens
                sleep_for = missing / (self.rate_per_minute / 60.0)
            await asyncio.sleep(sleep_for)


_BUCKET = TokenBucket(
    rate_per_minute=int(os.environ.get("RATE_LIMIT_RPM", "30")),
    burst=int(os.environ.get("RATE_LIMIT_BURST", "5")),
)


async def apply_limiter() -> None:
    """Apply the global outbound-call limiter."""
    await _BUCKET.acquire()
