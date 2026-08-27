"""OTel export configuration for the MAF handoff bridge, pointed at a Phoenix collector.

Two independent switches govern what Phoenix receives. `PHOENIX_COLLECTOR_ENDPOINT` decides
whether spans are exported at all; `ENABLE_SENSITIVE_DATA` (read by agent_framework itself)
decides whether those spans carry message content. Both are set by docker-compose.yml.

Configured entirely from environment variables so the same image works standalone (no
collector, spans just aren't exported) and wired into the docker-compose dev environment
(WS5), which sets `PHOENIX_COLLECTOR_ENDPOINT` to point at the sibling Phoenix container.
Phoenix remains a passive telemetry consumer only — nothing here calls back into the app.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_configured = False


def configure_telemetry() -> None:
    """Idempotently wire up `openinference-instrumentation-agent-framework` if a collector
    endpoint is configured. No-ops (does not raise) when the env var is unset, so running the
    server standalone for `curl localhost:PORT/agents/manifest` never depends on Phoenix.
    """

    global _configured
    if _configured:
        return
    _configured = True

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return

    from openinference.instrumentation.agent_framework import AgentFrameworkToOpenInferenceProcessor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "remote-maf-handoff-bridge")})
    provider = TracerProvider(resource=resource)
    # Order matters: `AgentFrameworkToOpenInferenceProcessor` rewrites each span's attributes
    # from agent-framework's native GenAI semconv into OpenInference's on `on_end`, so it must
    # run before the batch/export processor picks up the (now-mutated) span to send to Phoenix.
    # agent_framework's own instrumentation is enabled by default and emits spans via whatever
    # TracerProvider `trace.set_tracer_provider` installs below — there is no separate
    # `Instrumentor().instrument(...)` call for it (unlike openinference's OpenAI/LangChain
    # instrumentors); the package here only ships this span-attribute-translating processor.
    provider.add_span_processor(AgentFrameworkToOpenInferenceProcessor())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    # Message content (prompts, completions, tool arguments, tool results) is gated behind
    # agent_framework's `enable_sensitive_data` setting, which defaults to False — with it off
    # Phoenix renders the trace tree but every message pane is empty. The setting is read once
    # into a module-level singleton when `agent_framework` is first imported, so it can only be
    # turned on via the environment before startup, never from here. Warn rather than fail: a
    # metadata-only trace is still useful, but a silently empty Phoenix is not.
    from agent_framework.observability import OBSERVABILITY_SETTINGS

    if not OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED:
        logger.warning(
            "Exporting traces to %s without message content: set ENABLE_SENSITIVE_DATA=true "
            "before starting the process to record prompts, completions and tool calls.",
            endpoint,
        )
