"""
observability/tracing.py
 
Wires OpenTelemetry spans into the harness purely through lifecycle_hooks
-- this module never imports execution_loop.py or reaches into harness
internals directly. That's deliberate: tracing is an observer, not a
dependency of the harness, so the harness has zero knowledge that tracing
exists, and instrumentation can be added, swapped, or removed without
touching harness/ at all.
 
One span per LLM call, one span per tool call, correlated under a parent
span for the whole session/turn. Exports to console by default (for local
dev) or to an OTLP collector if OTEL_EXPORTER_OTLP_ENDPOINT is set, so the
same instrumentation code works locally and once deployed behind a real
collector (Honeycomb, Grafana Tempo, etc.) without code changes.
"""

from __future__ import annotations

import logging 
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass , field
from typing import Any , Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span , Status , StatusCode
from harness.core.lifecycle_hooks import HookContext , HookEvent , LifecycleHooks

logger = logging.getLogger("harborline.observability.tracing")

SERVICE_NAME = "harborline-agent-harness"

def configure_tracing(*,service_name: str = SERVICE_NAME , console:bool | None = None) -> trace.Tracer:
    """
    Sets up the global TracerProvider once per process. Call this at
    startup (before any harness runs) -- calling it more than once is
    harmless but redundant, OpenTelemetry keeps only the first provider
    that gets set as the global.
 
    `console`, if left None, is auto-decided: True (print spans to
    stdout) when no OTLP endpoint is configured, so local dev/testing has
    visible output without needing a collector running.
    """

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    use_console = console if console is not None else not otlp_endpoint

    provider = TracerProvider(resource=Resource.create({"service.name":service_name}))

    if use_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

@dataclass
class TracingMetricsSink:
    """
    In addition to emitting spans, this keeps an in-process rolling log of
    completed calls (model + tool) as plain dicts, which is what
    observability/dashboard/app.py reads to render cost/latency/pass-rate
    without needing to stand up a real trace backend just to see a chart.
    Bounded by `max_entries` so a long-running process doesn't leak memory.
    """

    max_entries: int = 5000
    entries: list[dict[str , Any]] = field(default_factory=list)

    def record(self , entry: dict[str , Any]) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def recent(self , limit: int = 100) -> list[dict[str, Any]]:
        return self.entries[-limit:]

class HarnessTracer:
    """
    Attaches OpenTelemetry instrumentation to a harness session's
    LifecycleHooks. One HarnessTracer per session (or share one across
    sessions if you want a single running metrics sink for a whole
    process -- see the dashboard, which does exactly that).
    """

    def __init__(self , hooks: LifecycleHooks , * , tracer:trace.Tracer | None = None , sink : TracingMetricsSink | None = None):
        self.hooks = hooks
        self.tracer = tracer or trace.get_tracer(SERVICE_NAME)
        self.sink = sink or TracingMetricsSink()
        self._open_spans : dict[str , tuple[Span , float]] = {}
        self._register()

    def _register(self) -> None:
        self.hooks.register(HookEvent.ON_MODEL_CALL, self._on_model_call, name="tracing.on_model_call")
        self.hooks.register(HookEvent.ON_MODEL_RESPONSE, self._on_model_response, name="tracing.on_model_response")
        self.hooks.register(HookEvent.ON_TOOL_CALL, self._on_tool_call, name="tracing.on_tool_call")
        self.hooks.register(HookEvent.ON_TOOL_RESULT, self._on_tool_result, name="tracing.on_tool_result")
        self.hooks.register(HookEvent.ON_ERROR, self._on_error, name="tracing.on_error")

    def _span_key(self , session_id: str , turn : int , kind: str) -> str:
        return f"{session_id}:{turn}:{kind}"

    def _on_model_call(self , ctx : contextmanager) -> None:
        span = self.tracer.start_span(
            "llm.call",
            attributes={
                "harborline.session_id": ctx.session_id,
                "harborline.turn": ctx.turn,
                "harborline.prompt_chars": len(ctx.data.get("prompt", "")),
            },
        )
        self._open_spans[self._span_key(ctx.session_id , ctx.turn , "llm")] = (span , time.monotonic())

    def _on_model_response(self, ctx: HookContext) -> None:
        key = self._span_key(ctx.session_id, ctx.turn, "llm")
        opened = self._open_spans.pop(key, None)
        response = ctx.data.get("response")
        duration_s = time.monotonic() - opened[1] if opened else None
 
        if opened:
            span, _ = opened
            if response is not None:
                span.set_attribute("harborline.model", getattr(response, "model", "unknown"))
                span.set_attribute("harborline.prompt_tokens", getattr(response, "prompt_tokens", 0))
                span.set_attribute("harborline.completion_tokens", getattr(response, "completion_tokens", 0))
                span.set_attribute("harborline.cost_usd", getattr(response, "cost_usd", 0.0))
                span.set_attribute("harborline.tool_calls_requested", len(getattr(response, "tool_calls", [])))
            span.set_status(Status(StatusCode.OK))
            span.end()
 
        self.sink.record({
            "type": "llm_call",
            "session_id": ctx.session_id,
            "turn": ctx.turn,
            "timestamp": ctx.timestamp,
            "duration_s": duration_s,
            "model": getattr(response, "model", "unknown") if response else "unknown",
            "prompt_tokens": getattr(response, "prompt_tokens", 0) if response else 0,
            "completion_tokens": getattr(response, "completion_tokens", 0) if response else 0,
            "cost_usd": getattr(response, "cost_usd", 0.0) if response else 0.0,
        })
 
    def _on_tool_call(self, ctx: HookContext) -> None:
        tool_name = ctx.data.get("tool_name", "unknown")
        span = self.tracer.start_span(
            f"tool.{tool_name}",
            attributes={
                "harborline.session_id": ctx.session_id,
                "harborline.turn": ctx.turn,
                "harborline.tool_name": tool_name,
                "harborline.call_id": ctx.data.get("call_id", ""),
            },
        )
        self._open_spans[self._span_key(ctx.session_id, ctx.turn, f"tool:{ctx.data.get('call_id')}")] = (
            span, time.monotonic()
        )
 
    def _on_tool_result(self, ctx: HookContext) -> None:
        result = ctx.data.get("result")
        call_id = getattr(result, "tool_name", None)
        # Look up by tool_name+call structure -- ON_TOOL_CALL and
        # ON_TOOL_RESULT share turn/session but the call_id lives on the
        # ToolResult here, not in ctx.data directly, so match on tool_name
        # within this turn.
        key = self._span_key(ctx.session_id, ctx.turn, f"tool:{ctx.data.get('tool_name')}")
        matched_key = None
        for k in self._open_spans:
            if k.startswith(f"{ctx.session_id}:{ctx.turn}:tool:"):
                matched_key = k
                break
        opened = self._open_spans.pop(matched_key, None) if matched_key else None
        duration_s = time.monotonic() - opened[1] if opened else None
 
        if opened and result is not None:
            span, _ = opened
            span.set_attribute("harborline.succeeded", getattr(result, "succeeded", False))
            span.set_attribute("harborline.attempts", getattr(result, "attempts", 1))
            if getattr(result, "succeeded", False):
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, str(getattr(result, "error", "tool failed"))))
            span.end()
 
        self.sink.record({
            "type": "tool_call",
            "session_id": ctx.session_id,
            "turn": ctx.turn,
            "timestamp": ctx.timestamp,
            "duration_s": duration_s,
            "tool_name": getattr(result, "tool_name", ctx.data.get("tool_name", "unknown")),
            "succeeded": getattr(result, "succeeded", None),
            "attempts": getattr(result, "attempts", None),
        })
 
    def _on_error(self, ctx: HookContext) -> None:
        self.sink.record({
            "type": "error",
            "session_id": ctx.session_id,
            "turn": ctx.turn,
            "timestamp": ctx.timestamp,
            "stage": ctx.data.get("stage"),
            "error": ctx.data.get("error"),
        })
        logger.error("harness error recorded: session=%s turn=%s %s", ctx.session_id, ctx.turn, ctx.data)
 
 
@contextmanager
def trace_eval_run(dataset_path: str, *, tracer: trace.Tracer | None = None) -> Generator[Span, None, None]:
    """
    Wraps an entire evaluate() call (see eval/runner.py) in a parent span,
    so every individual case's spans (via HarnessTracer, if the agent
    factory attaches one) nest under a single "eval.run" span for that
    dataset instead of appearing as unrelated top-level spans.
    """
    tracer = tracer or trace.get_tracer(SERVICE_NAME)
    with tracer.start_as_current_span("eval.run", attributes={"harborline.dataset_path": dataset_path}) as span:
        yield span