"""Optional Langfuse tracing for the ADK agents (graceful, best-effort).

Uses the OpenInference **Google-ADK instrumentor**: one ``init_tracing()`` call
per process auto-captures every ADK agent run, tool call, and model completion as
an OpenTelemetry span and forwards it to Langfuse — a framework integration, not
manual spans (Langfuse's recommended approach).

Enable by installing the ``observability`` extra and setting, in
``finance_coordinator/.env``::

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com     # or us.cloud… / self-hosted

**Graceful:** if the keys are unset or the packages aren't installed,
``init_tracing()`` is a silent no-op — agents run untraced and nothing breaks.

Call ``init_tracing()`` once **after** loading env (``load_dotenv``), and
``flush()`` before a short-lived process exits so buffered spans are sent.
"""

from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger("a2a_finance.observability")

_initialized = False
_provider = None  # the OTel TracerProvider we export from (for flush())


def configured() -> bool:
    """True if Langfuse credentials are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _base_url() -> str:
    return (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
            or "https://cloud.langfuse.com").rstrip("/")


def init_tracing() -> bool:
    """Initialize OTLP export to Langfuse + ADK instrumentation. Returns True if active.

    Uses the OpenTelemetry path: a TracerProvider with an OTLP HTTP exporter
    pointed at Langfuse's `/api/public/otel` endpoint, then the OpenInference ADK
    instrumentor bound to that provider. This is version-robust (works with the
    Langfuse v4 SDK, whose auto-tracer setup does not reliably record spans).

    Best-effort and idempotent: no-op (returns False) when credentials are missing
    or the ``observability`` extra isn't installed.
    """
    global _initialized, _provider
    if _initialized:
        return True
    if not configured():
        return False

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    except ImportError as e:
        logger.info("observability extra not installed (%s); "
                    "tracing disabled (uv sync --extra observability).", e)
        return False

    try:
        auth = base64.b64encode(
            f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
        ).decode()
        exporter = OTLPSpanExporter(
            endpoint=f"{_base_url()}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )
        # Order-independent: OTel allows the global TracerProvider to be set only
        # once, and google.adk may set/grab it before we run. If a real SDK
        # provider is already global, ATTACH our Langfuse exporter to it (so ADK's
        # spans export); otherwise create and register one.
        current = otel_trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            provider = current
        else:
            provider = TracerProvider()
            otel_trace.set_tracer_provider(provider)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        GoogleADKInstrumentor().instrument(tracer_provider=provider)
        _provider = provider
        _initialized = True
        logger.info("Langfuse tracing enabled (OTLP -> %s).", _base_url())
        return True
    except Exception as e:
        logger.warning("Langfuse init failed (tracing disabled): %s", e)
        return False


def flush() -> None:
    """Flush buffered spans — call before a short-lived process exits. No-op if off."""
    if not _initialized or _provider is None:
        return
    try:
        _provider.force_flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)
