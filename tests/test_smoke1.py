"""
End-to-end test: loads the golden dataset, spins up a real harness
instance per test case (same components from harness/), runs evaluate(),
then runs gate.py against the resulting report -- proving runner.py and
gate.py actually interoperate with harness/ and with each other, not just
that each file imports cleanly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from eval.gate import GatePolicy, check_gate
from eval.runner import HarnessInstance, default_grader_registry, evaluate, load_golden_dataset
from eval.types import TestCase
from harness.core.context_manager import ContextManager, ContextSource, SourceKind
from harness.core.execution_loop import ExecutionLoop, ModelResponse, ToolCallRequest
from harness.core.lifecycle_hooks import LifecycleHooks
from harness.core.state_store import StateStore
from harness.core.tool_registry import ToolRegistry


class SearchArgs(BaseModel):
    query: str


async def fake_search(query: str) -> str:
    return "Refunds are available within 30 days of purchase, no exceptions after that."


class ScriptedFakeModel:
    """
    A fake ModelClient that behaves like a real refund-policy agent: on
    its first call for a session it always calls search_docs, then
    answers based on the (simulated) tool result on the next call. Tracks
    turn number explicitly rather than inspecting prompt text, since the
    context manager pre-seeds retrieved context regardless of whether a
    tool has actually been called yet -- checking prompt content for that
    would be a false signal.
    """

    def __init__(self):
        self.turn = 0

    async def complete(self, prompt: str, tool_schemas):
        self.turn += 1
        if self.turn == 1:
            return ModelResponse(
                content="", tool_calls=[ToolCallRequest("search_docs", {"query": "refund policy"}, "call_1")],
                prompt_tokens=40, completion_tokens=10, cost_usd=0.0005, model="fake-v1",
            )
        return ModelResponse(
            content="You can get a refund within 30 days of purchase. No, refunds are not available after that window.",
            tool_calls=[], prompt_tokens=60, completion_tokens=20, cost_usd=0.0008, model="fake-v1",
        )


class FakeJudgeClient:
    """Deterministic fake judge so the test doesn't need a live API key either."""

    async def complete_text(self, prompt: str) -> str:
        if "30-day" in prompt or "30 days" in prompt:
            return '{"reasoning": "mentions the 30 day window correctly", "score": 0.9}'
        return '{"reasoning": "response is reasonably concise", "score": 0.85}'


def agent_factory(test_case: TestCase) -> HarnessInstance:
    registry = ToolRegistry()
    registry.register("search_docs", SearchArgs, description="search internal docs")(fake_search)

    context = ContextManager(token_budget=4000)
    context.set_system_prompt("You are a helpful support agent. Always search docs before answering policy questions.")
    context.add_source(ContextSource(
        source_ref=test_case.context_snapshot or "docs/refund-policy.md@v3",
        kind=SourceKind.RETRIEVED,
        content="Refunds are available within 30 days of purchase, no exceptions after that.",
    ))

    hooks = LifecycleHooks()
    loop = ExecutionLoop(
        model_client=ScriptedFakeModel(),
        tool_registry=registry,
        context_manager=context,
        state_store=StateStore(max_turns=5),
        hooks=hooks,
        max_turns=5,
    )
    return HarnessInstance(loop=loop, hooks=hooks)


async def main() -> None:
    dataset = load_golden_dataset("eval/datasets/golden/refund_policy.yaml")
    assert len(dataset) == 3, f"expected 3 cases, loaded {len(dataset)}"
    print(f"loaded {len(dataset)} test cases: {[c.id for c in dataset]}")

    registry = default_grader_registry(judge_client=FakeJudgeClient())
    report = await evaluate(dataset, agent_factory, registry, dataset_path="eval/datasets/golden/refund_policy.yaml")

    print("\nPer-case results:")
    for case in report.case_results:
        grade_str = ", ".join(f"{g.grader_name}={g.status.value}" for g in case.grades)
        print(f"  {case.test_case_id}: passed={case.passed} [{grade_str}]")

    print(f"\n{report.summary()}")

    assert report.case_results[0].passed, "rag-001 (contains '30 days') should pass"
    assert report.case_results[2].passed, "rag-003 (llm_judge) should pass"

    # Now run the gate -- this is the CI-blocking check.
    strict_policy = GatePolicy(min_pass_rate=1.0)
    decision = check_gate(report, strict_policy)
    print("\n--- gate check (strict) ---")
    print(decision.render())
    assert decision.passed, "gate should pass when all cases pass"

    # Prove the gate actually blocks: simulate a report with one failing case.
    from eval.runner import CaseResult
    from eval.types import GradeResult, GradeStatus
    from harness.core.execution_loop import TurnEndReason

    broken_report = report
    broken_report.case_results = broken_report.case_results + [
        CaseResult(
            test_case_id="rag-broken", end_reason=TurnEndReason.FINAL_ANSWERS, final_answer="wrong",
            grades=[GradeResult(grader_name="contains", status=GradeStatus.FAIL, score=0.0, reasoning="missing required text")],
            cost_usd=0.001, latency_s=0.5, tools_called=[],
        )
    ]
    decision2 = check_gate(broken_report, strict_policy)
    print("\n--- gate check (with injected failing case) ---")
    print(decision2.render())
    assert not decision2.passed, "gate should FAIL when a case fails and policy requires 100% pass rate"

    # And prove the allow-list override works.
    lenient_policy = GatePolicy(min_pass_rate=1.0, allowed_failing_case_ids=frozenset({"rag-broken"}))
    decision3 = check_gate(broken_report, lenient_policy)
    print("\n--- gate check (with allow-listed failing case) ---")
    print(decision3.render())
    assert decision3.passed, "gate should PASS when the only failing case is allow-listed"

    print("\nAll eval-layer smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())