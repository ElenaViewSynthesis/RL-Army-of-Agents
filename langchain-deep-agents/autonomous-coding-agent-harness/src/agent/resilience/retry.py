"""Exponential backoff retry helpers."""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agent.errors import RateLimitExceeded, RequestTooLarge, RetryExhausted

T = TypeVar("T")

_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", "3"))
_BASE_DELAY = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))
_MAX_DELAY = float(os.environ.get("RETRY_MAX_DELAY", "30.0"))


def _status_code(error: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _model_name(error: BaseException) -> str:
    return str(getattr(error, "model", "unknown"))


def normalize_provider_error(error: BaseException) -> BaseException:
    """Map common provider errors into typed agent errors."""
    status = _status_code(error)
    text = str(error).lower()
    if status == 413 or "request too large" in text or "too many tokens" in text:
        return RequestTooLarge(model=_model_name(error))
    if status == 429 and ("daily" in text or "quota" in text or "token cap" in text):
        return RateLimitExceeded(model=_model_name(error))
    return error


def _retryable(error: BaseException) -> bool:
    if isinstance(error, (RateLimitExceeded, RequestTooLarge)):
        return False
    status = _status_code(error)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return isinstance(error, (TimeoutError, ConnectionError))


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = _MAX_ATTEMPTS,
    base_delay: float = _BASE_DELAY,
    max_delay: float = _MAX_DELAY,
) -> T:
    """Run an async operation with exponential backoff."""
    attempts = max(max_attempts, 1)
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except BaseException as exc:
            normalized = normalize_provider_error(exc)
            if normalized is not exc:
                raise normalized from exc
            last_error = exc
            if attempt >= attempts or not _retryable(exc):
                raise RetryExhausted(attempt, exc) from exc
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay)

    raise RetryExhausted(attempts, last_error or RuntimeError("unknown retry failure"))
