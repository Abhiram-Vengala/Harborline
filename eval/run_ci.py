"""
eval/run_ci.py

The CI-facing entry point: builds an agent_factory, runs evaluate() against
a golden dataset, and writes the resulting report to disk as JSON --
that's the file ci/github-actions/eval-gate.yml hands to eval/gate.py.

Separate from eval/runner.py on purpose: runner.py is the library
(evaluate(), used the same way in a notebook, a script, or CI), this file
is the CI wiring around it -- constructing a specific agent_factory against
a specific provider, parsing CLI args, and serializing the result. Local/
dev use should import eval.runner directly rather than going through this
CLI, since agent_factory here is a fixed example wiring meant to be
replaced with your actual agent once harness/agents/ has one.

Two providers:
  --provider groq  (default) -- real model calls via Groq's free tier,
                                 needs GROQ_API_KEY (no credit card required,
                                 get one at https://console.groq.com/keys)
  --provider fake              -- deterministic canned responses, for
                                    proving the CI plumbing itself works
                                    without hitting Groq's rate limits or
                                    needing a key. Never use this to
                                    actually gate a merge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from eval.runner import HarnessInstance, default_grader_registry, evaluate, load_golden_dataset
from eval.types import TestCase
from harness.core.context_manager import ContextManager, ContextSource, SourceKind
from harness.core.execution_loop import ExecutionLoop, ModelResponse, ToolCallRequest
from harness.core.lifecycle_hooks import LifecycleHooks
from harness.core.state_store import StateStore
from harness.core.tool_registry import ToolRegistry

load_dotenv()

# Placeholder document store for the example search_docs tool. Replace
# with a real retriever once harness/agents/ has an actual agent -- this
# exists so the CI pipeline has something concrete to run end to end out
# of the box.
_DOCS = {
    "refund-policy": "Refunds are available within 30 days of purchase. No refunds are issued after that window.",
}


class SearchArgs(BaseModel):
    query: str


async def _search_docs(query: str) -> str:
    for key, content in _DOCS.items():
        if key.replace("-", " ") in query.lower() or "refund" in query.lower():
            return content
    return "no matching documents found"


class FakeModelClient:
    """Deterministic stand-in for --provider fake. See module docstring."""

    def __init__(self):
        self._turn = 0

    async def complete(self, prompt: str, tool_schemas) -> ModelResponse:
        self._turn += 1
        if self._turn % 2 == 1:
            return ModelResponse(
                content="", tool_calls=[ToolCallRequest("search_docs", {"query": "refund policy"}, f"c{self._turn}")],
                prompt_tokens=30, completion_tokens=8, cost_usd=0.0001, model="fake",
            )
        return ModelResponse(
            content="Refunds are available within 30 days of purchase; no refunds after that.",
            tool_calls=[], prompt_tokens=50, completion_tokens=15, cost_usd=0.0002, model="fake",
        )


class FakeJudgeClient:
    async def complete_text(self, prompt: str) -> str:
        return '{"reasoning": "fake judge, plumbing test only", "score": 0.9}'


def _build_agent_factory(provider: str):
    if provider == "fake":
        def factory(test_case: TestCase) -> HarnessInstance:
            return _make_instance(test_case, FakeModelClient())
        return factory, FakeJudgeClient()

    if provider == "groq":
        from eval.providers.groq_client import GroqJudgeClient, GroqModelClient

        def factory(test_case: TestCase) -> HarnessInstance:
            return _make_instance(test_case, GroqModelClient())
        return factory, GroqJudgeClient()

    raise ValueError(f"unknown provider {provider!r}, expected 'groq' or 'fake'")


def _make_instance(test_case: TestCase, model_client) -> HarnessInstance:
    registry = ToolRegistry()
    registry.register("search_docs", SearchArgs, description="search internal knowledge base")(_search_docs)

    context = ContextManager(token_budget=4000)
    context.set_system_prompt(
        "You are a support agent. Use the search_docs tool before answering policy questions. "
        "Answer concisely based only on what search_docs returns."
    )
    if test_case.context_snapshot:
        context.add_source(ContextSource(
            source_ref=test_case.context_snapshot,
            kind=SourceKind.RETRIEVED,
            content=_DOCS.get("refund-policy", ""),
        ))

    hooks = LifecycleHooks()
    loop = ExecutionLoop(
        model_client=model_client,
        tool_registry=registry,
        context_manager=context,
        state_store=StateStore(max_turns=5),
        hooks=hooks,
        max_turns=5,
    )
    return HarnessInstance(loop=loop, hooks=hooks)


async def run(dataset_path: str, out_path: str, provider: str, concurrency: int) -> int:
    dataset = load_golden_dataset(dataset_path)
    agent_factory, judge_client = _build_agent_factory(provider)
    grader_registry = default_grader_registry(judge_client=judge_client)

    print(f"running {len(dataset)} test case(s) from {dataset_path} via provider={provider!r}...")
    report = await evaluate(
        dataset, agent_factory, grader_registry,
        concurrency=concurrency, dataset_path=dataset_path,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report.to_dict(), indent=2))

    print(report.summary())
    print(f"report written to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden dataset through the harness and write an eval report.")
    parser.add_argument("--dataset", required=True, help="Path to a golden dataset file or directory")
    parser.add_argument("--out", required=True, help="Path to write the JSON eval report")
    parser.add_argument("--provider", choices=["groq", "fake"], default="groq")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    if args.provider == "groq":
        import os
        if not os.environ.get("GROQ_API_KEY"):
            print(
                "ERROR: --provider groq requires GROQ_API_KEY to be set "
                "(as a repo secret in CI, or exported locally). Get a free key, "
                "no credit card needed, at https://console.groq.com/keys. "
                "Use --provider fake to test the pipeline without a key.",
                file=sys.stderr,
            )
            return 1

    return asyncio.run(run(args.dataset, args.out, args.provider, args.concurrency))


if __name__ == "__main__":
    sys.exit(main())