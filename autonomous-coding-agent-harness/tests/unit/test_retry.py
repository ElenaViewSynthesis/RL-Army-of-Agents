import asyncio

from agent.errors import RateLimitExceeded, RequestTooLarge, RetryExhausted
from agent.resilience.retry import normalize_provider_error, with_retry


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


def test_normalize_request_too_large() -> None:
    error = normalize_provider_error(ProviderError("request too large", 413))

    assert isinstance(error, RequestTooLarge)


def test_normalize_daily_rate_limit() -> None:
    error = normalize_provider_error(ProviderError("daily token quota exceeded", 429))

    assert isinstance(error, RateLimitExceeded)


def test_with_retry_succeeds_after_retryable_failure() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("temporarily unavailable", 503)
        return "ok"

    result = asyncio.run(with_retry(flaky, max_attempts=2, base_delay=0, max_delay=0))

    assert result == "ok"
    assert calls == 2


def test_with_retry_raises_retry_exhausted() -> None:
    async def failing() -> str:
        raise ProviderError("temporarily unavailable", 503)

    try:
        asyncio.run(with_retry(failing, max_attempts=2, base_delay=0, max_delay=0))
    except RetryExhausted as exc:
        assert exc.attempts == 2
        return

    raise AssertionError("retry exhaustion should raise")
