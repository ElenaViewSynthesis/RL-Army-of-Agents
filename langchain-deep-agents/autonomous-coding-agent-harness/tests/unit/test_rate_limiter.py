import asyncio

from agent.resilience.limiter import TokenBucket


def test_token_bucket_acquire_consumes_token() -> None:
    bucket = TokenBucket(rate_per_minute=60, burst=2)

    asyncio.run(bucket.acquire())

    assert bucket.tokens < 2


def test_token_bucket_has_minimum_capacity() -> None:
    bucket = TokenBucket(rate_per_minute=0, burst=0)

    assert bucket.rate_per_minute == 1
    assert bucket.capacity == 1
