"""
observability/dashboard/app.py

A minimal metrics dashboard: reads from a TracingMetricsSink (live harness
call data -- see observability/tracing.py) and from eval report JSON files
written by eval/runner.py, and serves both as JSON + a static HTML page
with charts. This is intentionally NOT a general-purpose observability
platform -- it's the "screenshot for your portfolio and a quick sanity
check while developing" dashboard the build plan calls for in week 6.
For anything beyond that, point a real OTLP collector at the traces this
module also emits (see tracing.py) and use a proper backend (Grafana,
Honeycomb, etc).

Run: uvicorn observability.dashboard.app:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from observability.tracing import TracingMetricsSink

STATIC_DIR = Path(__file__).parent / "static"
REPORTS_DIR = Path(__file__).parent.parent.parent / "eval" / "reports"

app = FastAPI(title="Harborline Observability Dashboard")

# A single process-wide sink. In a real deployment, HarnessTracer instances
# created per session should be constructed with this same sink (pass it
# explicitly) so live traffic shows up here rather than each session
# accumulating an isolated, invisible sink of its own.
metrics_sink = TracingMetricsSink(max_entries=10_000)


def get_metrics_sink() -> TracingMetricsSink:
    """
    Indirection point: swap this for a real store (Redis, a DB) behind the
    same interface if the dashboard needs to survive process restarts or
    run against multiple app instances. Routes below call this function
    rather than touching the module-level `metrics_sink` directly, so that
    swap is a one-line change.
    """
    return metrics_sink


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics/live")
def live_metrics(limit: int = 200) -> dict[str, Any]:
    """Recent LLM/tool calls and errors from the in-process tracing sink."""
    sink = get_metrics_sink()
    entries = sink.recent(limit=limit)

    llm_calls = [e for e in entries if e["type"] == "llm_call"]
    tool_calls = [e for e in entries if e["type"] == "tool_call"]
    errors = [e for e in entries if e["type"] == "error"]

    total_cost = sum(e.get("cost_usd", 0.0) or 0.0 for e in llm_calls)
    durations = [e["duration_s"] for e in llm_calls if e.get("duration_s") is not None]
    mean_latency = sum(durations) / len(durations) if durations else 0.0
    tool_success_rate = (
        sum(1 for e in tool_calls if e.get("succeeded")) / len(tool_calls) if tool_calls else None
    )

    return {
        "summary": {
            "llm_call_count": len(llm_calls),
            "tool_call_count": len(tool_calls),
            "error_count": len(errors),
            "total_cost_usd": total_cost,
            "mean_llm_latency_s": mean_latency,
            "tool_success_rate": tool_success_rate,
        },
        "entries": entries,
    }


@app.get("/api/metrics/eval-reports")
def eval_reports() -> dict[str, Any]:
    """
    Lists eval reports found in eval/reports/ (JSON files written by a CI
    job or a local `evaluate()` run via report.to_dict() +
    json.dump(...)), most recent first, with pass-rate/cost/latency
    trends across them -- this is what turns individual CI runs into a
    trend line instead of a series of disconnected pass/fail dots.
    """
    if not REPORTS_DIR.exists():
        return {"reports": [], "note": f"no reports directory at {REPORTS_DIR}"}

    reports = []
    for path in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        reports.append({
            "file": path.name,
            "modified": path.stat().st_mtime,
            "pass_rate": data.get("pass_rate"),
            "total_cost_usd": data.get("total_cost_usd"),
            "mean_latency_s": data.get("mean_latency_s"),
            "case_count": len(data.get("cases", [])),
        })
    reports.sort(key=lambda r: r["modified"])
    return {"reports": reports}


@app.get("/")
def dashboard_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")