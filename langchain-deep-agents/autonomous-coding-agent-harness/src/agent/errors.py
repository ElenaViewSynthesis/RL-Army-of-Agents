"""Typed error hierarchy for the agent harness."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for public agent errors."""


class RetryExhausted(AgentError):
    """Raised when retryable work fails after all attempts."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"retry exhausted after {attempts} attempts: {last_error}")


class RateLimitExceeded(AgentError):
    """Raised for non-retryable daily/token-cap rate limits."""

    def __init__(self, model: str = "unknown", retry_after: int | None = None) -> None:
        self.model = model
        self.retry_after = retry_after
        suffix = f" retry_after={retry_after}" if retry_after is not None else ""
        super().__init__(f"rate limit exceeded for model={model}{suffix}")


class RequestTooLarge(AgentError):
    """Raised when a request exceeds the model/provider size limit."""

    def __init__(self, model: str = "unknown", requested_tokens: int | None = None) -> None:
        self.model = model
        self.requested_tokens = requested_tokens
        suffix = f" requested_tokens={requested_tokens}" if requested_tokens is not None else ""
        super().__init__(f"request too large for model={model}{suffix}")


class ToolError(AgentError):
    """Raised when a tool fails at a public boundary."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"tool {tool_name!r} failed: {message}")


class RetrievalMiss(AgentError):
    """Raised when retrieval cannot provide useful tools."""

    def __init__(self, goal: str, k: int) -> None:
        self.goal = goal
        self.k = k
        super().__init__(f"retrieval miss at k={k}: {goal[:120]!r}")


class ValidationError(AgentError):
    """Raised for schema validation failures at agent-owned boundaries."""
