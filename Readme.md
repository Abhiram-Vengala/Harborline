# Harborline

A self-hostable agent harness with a built-in evaluation layer, merge-blocking
CI gate, observability, and cloud deployment — built to close the gap between
running an agent and running one you can trust in production.

Most "AI agent" projects stop at "it works when I run it." Harborline exists
to answer a different question: **how do you know it still works after the
next change, and how expensive/slow did that change make it?** That's the
part most portfolio projects skip, and it's the part production teams
actually get paid to solve.

## Why this exists

Agent harness engineering and eval-gated CI/CD have converged as a pattern
across several companies in 2026 independently — eval-first observability
platforms, YC-backed agent optimization tools, and open-source eval
frameworks are all solving overlapping pieces of the same problem. This
project is a synthesis of that pattern, not a clone of any one of them, plus
one deliberate addition most of them skip: **validating the *inputs* an
agent reasons from, not just scoring its outputs.** A model can produce a
perfectly-argued answer built on stale or malformed context, and a
pure output-grader will never catch that — it'll just give a high score to
a wrong answer. Harborline's `InputValidator` runs *before* the model is
even called.

## Architecture

```
harness/            The agent runtime
├── core/
│   ├── execution_loop.py    orchestrates turns: build context → call model →
│   │                        dispatch tool calls → repeat until final answer
│   ├── tool_registry.py     schema-validated tool calls, timeouts, retries
│   ├── context_manager.py   token-budgeted context assembly, source tracking
│   ├── state_store.py       conversation history, memory, cost accounting
│   └── lifecycle_hooks.py   pre/post-turn, tool-call, error event system
└── validation/
    └── input_validator.py   freshness + schema-drift checks, pre-flight

eval/                The evaluation layer
├── metrics/
│   ├── deterministic.py     exact-match, regex, JSON schema, tool-call checks
│   ├── llm_judge.py         G-Eval-style rubric scoring with retry/parsing safety
│   └── cost_latency.py      budget and regression-vs-baseline graders
├── datasets/golden/         versioned YAML test cases
├── providers/                model client adapters (currently: Groq)
├── runner.py                 evaluate(): runs a dataset through the harness
├── run_ci.py                 CI entry point: wires a provider, writes a report
└── gate.py                   merge-blocking pass/fail decision + CLI

observability/       What's happening in production
├── tracing.py        OpenTelemetry spans per LLM/tool call, via lifecycle hooks
└── dashboard/         FastAPI + Chart.js dashboard: cost, latency, pass-rate trends

ci/github-actions/eval-gate.yml   PR workflow: unit tests → real eval run → gate
deploy/               Dockerfile, docker-compose (local dev), render.yaml
tests/                End-to-end smoke tests for every layer above
```

Every piece above is wired through **protocols and hooks, not direct
imports** — `execution_loop.py` doesn't know Groq exists; it depends on a
`ModelClient` protocol. `tracing.py` doesn't import the execution loop; it
attaches to `LifecycleHooks`. That's what makes each piece independently
testable and swappable (see [providers](#swapping-model-providers) below).

## How a request actually flows

1. `ExecutionLoop.run(user_input)` starts a turn.
2. `ContextManager.build()` assembles system prompt + retrieved docs + memory
   into a token-budgeted context, each source tagged with a version hash.
3. `InputValidator.validate()` checks that context for staleness, schema
   drift, and completeness. **If it fails, the turn stops here — the model
   is never called on bad input.**
4. The model is called. If it asks for a tool, `ToolRegistry.execute()`
   validates the arguments against a pydantic schema, runs the tool with a
   timeout, and (if configured) retries on transient failure.
5. The result feeds back into context; loop repeats until the model returns
   a final answer, or `max_turns`/cost budget is hit.
6. `LifecycleHooks` fires at every step above — `tracing.py` listens and
   emits OpenTelemetry spans + records to the dashboard's metrics sink.

## Quickstart

```bash
git clone <your-repo-url> harborline && cd harborline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the harness + eval smoke tests (no API key needed — these use fake
model clients):

```bash
python tests/test_smoke.py
python tests/test_smoke1.py
```

Run the golden dataset against a real model (free, no credit card — get a
key at [console.groq.com/keys](https://console.groq.com/keys)):

```bash
# Loads GROQ_API_KEY from .env when present; exporting it also works.
export GROQ_API_KEY=your-key-here
python -m eval.run_ci --dataset eval/datasets/golden --out eval/reports/local.json --provider groq
python -m eval.gate --report eval/reports/local.json --min-pass-rate 0.9
```

The Groq adapter uses `openai/gpt-oss-120b` for the agent and
`openai/gpt-oss-20b` for judging by default. Set `GROQ_MODEL` or
`GROQ_JUDGE_MODEL` to select different models available to your Groq account.

On Windows PowerShell, `python -m eval.run_ci` loads `GROQ_API_KEY` from a
local `.env` file automatically after installing the requirements. To set it
for the current PowerShell session instead, use `$env:GROQ_API_KEY = "..."`.

Run the observability dashboard locally:

```bash
uvicorn observability.dashboard.app:app --reload --port 8000
# open http://localhost:8000
```

## Writing a golden test case

Test cases live in `eval/datasets/golden/*.yaml`. Each one names an input,
what the agent is expected to do, and which graders check the result:

```yaml
- id: rag-001
  input: "What's our refund policy for annual plans?"
  expected_tool_calls: ["search_docs"]
  context_snapshot: "docs/refund-policy.md@v3"
  max_cost_usd: 0.01
  max_latency_s: 8.0
  graders:
    - type: contains
      must_contain: ["30 days"]
    - type: tool_calls
    - type: cost_budget
    - type: latency_budget
```

Add new cases by appending, don't edit existing ones once they've run in CI
— change the `id` if expected behavior itself changes, so pass/fail history
per case stays meaningful over time.

## The CI gate

`.github/workflows/eval-gate.yml` (copy `ci/github-actions/eval-gate.yml`
there) runs on every PR touching `harness/` or `eval/`:

1. **Unit tests** — the fake-client smoke tests, fast, no external calls.
2. **Eval gate** — runs the real golden dataset against Groq, writes a
   report, then `eval/gate.py` decides pass/fail against configurable
   thresholds (`--min-pass-rate`, `--max-total-cost-usd`,
   `--max-mean-latency-s`). A failing gate fails the check and blocks merge.

Requires a `GROQ_API_KEY` repository secret.

## Swapping model providers

`eval/providers/groq_client.py` implements two small protocols
(`ModelClient` for the agent, `JudgeModelClient` for eval scoring). Neither
`harness/` nor `eval/metrics/llm_judge.py` import Groq directly — to switch
providers, write a new adapter implementing the same two methods
(`complete()` and `complete_text()`) and point `eval/run_ci.py`'s
`_build_agent_factory()` at it. Nothing else in the codebase changes.

## Deployment

```bash
docker build -f deploy/Dockerfile -t harborline .
docker run -p 8000:8000 -e GROQ_API_KEY=... harborline
```

Or for local iteration with hot reload: `docker compose -f deploy/docker-compose.yml up`.

`deploy/render.yaml` is a Render Blueprint for one-click deployment of the
dashboard service from the same Dockerfile.

## What's deliberately out of scope

- **No persistent storage** for state/memory — everything is in-process.
  Swapping `StateStore` for a Redis/Postgres-backed version is a natural
  next step, kept out here to keep the harness itself easy to read.
- **No multi-agent orchestration.** This is a single-agent harness. Composing
  several harness instances into a multi-agent system is a different, larger
  problem than what this project set out to prove.
- **The dashboard is not a production observability platform.** It's a
  "sanity check while developing + a screenshot for a portfolio" tool. Point
  `OTEL_EXPORTER_OTLP_ENDPOINT` at a real collector (Honeycomb, Grafana
  Tempo) for anything beyond that — `tracing.py` already emits standard
  OpenTelemetry spans, so this is a config change, not a rewrite.

## Status

Harness, eval layer, observability, CI gate, and deploy configs are built
and tested end to end with fake model/judge clients. Real-provider runs
(Groq) are wired but should be validated locally with a real API key before
being trusted in CI. See `tests/` for what's actually proven to work.