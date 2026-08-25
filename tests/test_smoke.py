"""
End-to-end smoke test wiring every harness component together with a fake
model client and fake tools, so the pieces are proven to actually
interoperate, not just unit-correct in isolation.

Run: python -m pytest tests/test_smoke.py -v
(or: python tests/test_smoke.py  -- it also runs standalone via __main__)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from harness.core.context_manager import ContextManager, ContextSource, SourceKind
from harness.core.execution_loop import (
    ExecutionLoop,
    ModelResponse,
    ToolCallRequest,
    TurnEndReason,
)
from harness.core.lifecycle_hooks import HookEvent, LifecycleHooks
from harness.core.state_store import StateStore
from harness.core.tool_registry import ToolRegistry
from harness.validation.input_validator import InputValidator, SourceRequirement


# ---------------------------------------------------------------------- #
# Fakes
# ---------------------------------------------------------------------- #

class ScriptedModelClient:
    """Replays a fixed sequence of ModelResponses, one per call."""

    def __init__(self, script: list[ModelResponse]):
        self._script = list(script)
        self.calls = 0

    async def complete(self, prompt: str, tool_schemas):
        self.calls += 1
        if not self._script:
            raise RuntimeError("script exhausted -- test asked for more turns than scripted")
        return self._script.pop(0)


class SearchArgs(BaseModel):
    query: str


async def fake_search(query: str) -> str:
    return f"3 results for {query!r}"


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search_docs", SearchArgs, description="search internal docs")(fake_search)
    return registry


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #

async def test_happy_path_with_tool_call():
    registry = build_registry()
    context = ContextManager(token_budget=4000)
    context.set_system_prompt("You are a helpful assistant.")
    context.add_source(ContextSource(
        source_ref="docs/refund-policy.md@v3",
        kind=SourceKind.RETRIEVED,
        content="Refunds are available within 30 days.",
        droppable=True,
    ))
    state = StateStore(max_turns=5)
    hooks = LifecycleHooks()

    fired = []
    hooks.register(HookEvent.ON_TOOL_CALL, lambda ctx: fired.append(ctx.event))

    model = ScriptedModelClient([
        ModelResponse(
            content="",
            tool_calls=[ToolCallRequest("search_docs", {"query": "refund"}, "call_1")],
            prompt_tokens=50, completion_tokens=10, cost_usd=0.001,
        ),
        ModelResponse(
            content="Refunds are available within 30 days of purchase.",
            tool_calls=[],
            prompt_tokens=80, completion_tokens=20, cost_usd=0.002,
        ),
    ])

    loop = ExecutionLoop(
        model_client=model,
        tool_registry=registry,
        context_manager=context,
        state_store=state,
        hooks=hooks,
        max_turns=5,
    )

    result = await loop.run("What's your refund policy?")

    assert result.end_reason == TurnEndReason.FINAL_ANSWERS
    assert result.final_answer is not None and "30 days" in result.final_answer
    assert model.calls == 2
    assert fired == [HookEvent.ON_TOOL_CALL]
    assert state.cost_summary().total_cost_usd > 0
    print("test_happy_path_with_tool_call: PASS")


async def test_validation_blocks_on_missing_required_source():
    registry = build_registry()
    context = ContextManager(token_budget=4000)
    context.set_system_prompt("You are a helpful assistant.")
    # Deliberately do NOT add the required source -- validator should block.

    validator = InputValidator()
    validator.add_requirement(SourceRequirement(source_ref="docs/refund-policy.md@v3", required=True))

    state = StateStore(max_turns=5)
    model = ScriptedModelClient([])  # should never be called

    loop = ExecutionLoop(
        model_client=model,
        tool_registry=registry,
        context_manager=context,
        state_store=state,
        input_validator=validator,
        max_turns=5,
        block_on_validation_error=True,
    )

    result = await loop.run("What's your refund policy?")

    assert result.end_reason == TurnEndReason.VALIDATION_BLOCKED
    assert result.validation_result is not None and result.validation_result.has_error
    assert model.calls == 0, "model should never be called when validation blocks the turn"
    print("test_validation_blocks_on_missing_required_source: PASS")


async def test_max_turns_exceeded_without_final_answer():
    registry = build_registry()
    context = ContextManager(token_budget=4000)
    context.set_system_prompt("You are a helpful assistant.")
    state = StateStore(max_turns=2)

    # Model always asks for another tool call, never finalizes.
    def never_ending():
        while True:
            yield ModelResponse(
                content="",
                tool_calls=[ToolCallRequest("search_docs", {"query": "x"}, "call_x")],
                prompt_tokens=10, completion_tokens=5, cost_usd=0.0001,
            )

    class InfiniteModel:
        def __init__(self):
            self._gen = never_ending()

        async def complete(self, prompt, tool_schemas):
            return next(self._gen)

    loop = ExecutionLoop(
        model_client=InfiniteModel(),
        tool_registry=registry,
        context_manager=context,
        state_store=state,
        max_turns=2,
    )

    result = await loop.run("loop forever please")

    assert result.end_reason == TurnEndReason.MAX_TURNS
    assert result.final_answer is None
    print("test_max_turns_exceeded_without_final_answer: PASS")


async def test_tool_execution_error_is_reported_not_raised():
    registry = ToolRegistry()

    class FailArgs(BaseModel):
        x: int

    async def always_fails(x: int) -> str:
        raise ValueError("simulated downstream failure")

    registry.register("flaky_tool", FailArgs, retryable=True, max_retries=1, retry_backoff_s=0.01)(always_fails)

    context = ContextManager(token_budget=4000)
    context.set_system_prompt("sys")
    state = StateStore(max_turns=5)

    model = ScriptedModelClient([
        ModelResponse(
            content="", tool_calls=[ToolCallRequest("flaky_tool", {"x": 1}, "call_1")],
            prompt_tokens=5, completion_tokens=5, cost_usd=0.0,
        ),
        ModelResponse(content="Handled the tool failure gracefully.", tool_calls=[],
                       prompt_tokens=5, completion_tokens=5, cost_usd=0.0),
    ])

    loop = ExecutionLoop(
        model_client=model, tool_registry=registry, context_manager=context,
        state_store=state, max_turns=5,
    )
    result = await loop.run("try the flaky tool")

    assert result.end_reason == TurnEndReason.FINAL_ANSWERS
    history = state.get_history()
    tool_messages = [m for m in history if m.tool_name == "flaky_tool"]
    assert len(tool_messages) == 1
    assert "ERROR" in tool_messages[0].content
    print("test_tool_execution_error_is_reported_not_raised: PASS")


async def main():
    await test_happy_path_with_tool_call()
    await test_validation_blocks_on_missing_required_source()
    await test_max_turns_exceeded_without_final_answer()
    await test_tool_execution_error_is_reported_not_raised()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())