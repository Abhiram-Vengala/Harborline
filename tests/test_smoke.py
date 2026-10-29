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

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic import BaseModel

from harness.core.context_manager import ContextManager, ContextSource, SourceKind
from harness.core.execution_loop import (
    ExecutionLoop,
    ModelResponse,
    ToolCallRequest,
    TurnEndReason,
)
from harness.core.lifecycle_hooks import HookEvent, LifecycleHooks , HookContext
from harness.core.state_store import StateStore , SnapshotNotFoundError , InvalidRoleError , StateStoreError
from harness.core.tool_registry import ToolRegistry
from harness.validation.input_validator import InputValidator, SourceRequirement
from harness.core.state_store import Message , Role
from harness.core.tool_registry import ToolRegistry , ToolResult
from observability.tracing import HarnessTracer


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

    assert result.end_reason == TurnEndReason.FINAL_ANSWER
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

    assert result.end_reason == TurnEndReason.FINAL_ANSWER
    history = state.get_history()
    tool_messages = [m for m in history if m.tool_name == "flaky_tool"]
    assert len(tool_messages) == 1
    assert "ERROR" in tool_messages[0].content
    print("test_tool_execution_error_is_reported_not_raised: PASS")

class RecordingModelClient:
    """Records every prompt it receives and replays scripted responses."""
    def __init__(self, script):
        self._script = list(script)
        self.prompts: list[str] = []

    async def complete(self, prompt, tool_schemas):
        self.prompts.append(prompt)
        return self._script.pop(0)


async def test_tool_results_reach_second_prompt():
    sentinel = "SENTINEL-7f3a9c"

    class Args(BaseModel):
        query: str

    async def lookup(query: str) -> str:
        return f"{sentinel}: refunds are available within 30 days"

    registry = ToolRegistry()
    registry.register("lookup", Args, description="lookup")(lookup)

    context = ContextManager(token_budget=4000)
    context.set_system_prompt("You are a helpful assistant.")
    model = RecordingModelClient([
        ModelResponse(content="", tool_calls=[ToolCallRequest("lookup", {"query": "refund"}, "call_1")]),
        ModelResponse(content="Done.", tool_calls=[]),
    ])
    loop = ExecutionLoop(
        model_client=model, tool_registry=registry, context_manager=context,
        state_store=StateStore(max_turns=5), max_turns=5,
    )

    question = "What is the refund window for annual plans?"
    result = await loop.run(question)

    assert result.end_reason == TurnEndReason.FINAL_ANSWER
    assert len(model.prompts) == 2
    assert question in model.prompts[0], "user question missing from first prompt"
    assert sentinel not in model.prompts[0]
    assert sentinel in model.prompts[1], "tool result missing from second prompt"
    assert model.prompts[0] != model.prompts[1]
    print("test_tool_results_reach_second_prompt: PASS")


def test_history_budget_pins_latest_user_message():
    cm = ContextManager(token_budget=200, token_counter=len)  # 1 char = 1 token
    cm.set_system_prompt("sys")
    history = [
        Message(role=Role.USER, content="task"),
        Message(role=Role.ASSISTANT, content="x" * 190),
        Message(role=Role.ASSISTANT, content="recent"),
    ]
    assembled = cm.build(history=history)
    assert [m.content for m in assembled.history] == ["task", "recent"]
    assert assembled.dropped_history_count == 1
    assert assembled.estimated_tokens <= 200
    print("test_history_budget_pins_latest_user_message: PASS")

async def test_registry_returns_errors_instead_of_raising():
    registry = build_registry()   # search_docs(query: str)
    r1 = await registry.execute("does_not_exist", {})
    assert not r1.succeeded and r1.error_type == "tool_not_found" and "search_docs" in r1.error
    r2 = await registry.execute("search_docs", {"wrong": 1})
    assert not r2.succeeded and r2.error_type == "invalid_arguments" and "query" in r2.error
    print("test_registry_returns_errors_instead_of_raising: PASS")


async def test_bad_tool_calls_are_fed_back_to_model():
    context = ContextManager(token_budget=4000)
    context.set_system_prompt("sys")
    state = StateStore(max_turns=5)
    model = RecordingModelClient([
        ModelResponse(content="", tool_calls=[ToolCallRequest("does_not_exist", {}, "c1")]),
        ModelResponse(content="", tool_calls=[ToolCallRequest("search_docs", {"wrong": 1}, "c2")]),
        ModelResponse(content="recovered", tool_calls=[]),
    ])
    loop = ExecutionLoop(
        model_client=model, tool_registry=build_registry(), context_manager=context,
        state_store=state, max_turns=5,
    )
    result = await loop.run("hi")

    assert result.end_reason == TurnEndReason.FINAL_ANSWER
    assert "does_not_exist" in model.prompts[1] and "search_docs" in model.prompts[1]
    assert "query" in model.prompts[2]
    tool_msgs = [m for m in state.get_history() if m.role == Role.TOOL]
    assert len(tool_msgs) == 2 and all(m.metadata.get("error") for m in tool_msgs)
    print("test_bad_tool_calls_are_fed_back_to_model: PASS")

def test_state_store_round_trip():
    store = StateStore(session_id="s1", max_turns=7)
    store.add_message(Role.USER, "hi")
    store.add_message(Role.TOOL, "result", tool_name="search_docs", tool_call_id="c1", metadata={"error": True})
    store.set_memory("k", {"a": 1})
    store.increment_turn()
    store.record_cost("m", 10, 5, 0.01)

    restored = StateStore.from_json(store.to_json())
    assert restored.session_id == "s1"
    assert restored.max_turns == 7
    assert restored.turns_remaining() == 6
    assert restored.get_memory("k") == {"a": 1}
    assert [m.to_dict() for m in restored.get_history()] == [m.to_dict() for m in store.get_history()]
    assert restored.cost_summary().total_cost_usd == store.cost_summary().total_cost_usd

    legacy = store.to_dict()
    legacy["max_truns"] = legacy.pop("max_turns")
    assert StateStore.from_dict(legacy).max_turns == 7
    print("test_state_store_round_trip: PASS")

def test_snapshot_restore_and_diff():
    store = StateStore(max_turns=10)
    store.add_message(Role.USER, "task")
    store.set_memory("plan", {"steps": ["a"]})
    store.increment_turn()
    store.snapshot("before")

    store.add_message(Role.ASSISTANT, "did stuff")
    store.get_memory("plan")["steps"].append("b")     # nested mutation after the snapshot
    store.set_memory("extra", 1)
    store.increment_turn()
    store.record_cost("m", 10, 5, 0.01)

    d = store.diff("before")
    assert [m.content for m in d.messages_added] == ["did stuff"]
    assert d.memory_added == {"extra": 1}
    assert "plan" in d.memory_changed
    assert d.turn_delta == 1 and abs(d.cost_delta_usd - 0.01) < 1e-9

    store.restore("before")
    assert [m.content for m in store.get_history()] == ["task"]
    assert store.get_memory("plan") == {"steps": ["a"]}   # fails with the old shallow copy
    assert "extra" not in store.memory_keys()
    assert store.turn_count == 2                           # budgets survive restore
    assert store.cost_summary().total_cost_usd > 0

    store.restore("before", restore_budgets=True)
    assert store.turn_count == 1 and store.cost_summary().total_cost_usd == 0
    assert store.diff("before").is_empty

    try:
        store.restore("nope")
    except SnapshotNotFoundError:
        pass
    else:
        raise AssertionError("expected SnapshotNotFoundError")
    print("test_snapshot_restore_and_diff: PASS")

def test_get_memory_defaults():
    store = StateStore()
    assert store.get_memory("missing") is None
    assert store.get_memory("missing", "fallback") == "fallback"
    store.set_memory("zero", 0)
    store.set_memory("empty", "")
    assert store.get_memory("zero", "fallback") == 0       # stored falsy values are returned as-is
    assert store.get_memory("empty", "fallback") == ""
    print("test_get_memory_defaults: PASS")

def test_invalid_role_is_a_state_store_error():
    assert issubclass(InvalidRoleError, StateStoreError)
    store = StateStore()
    try:
        store.add_message("wizard", "hi")
    except StateStoreError as e:           # a broad catch must now work
        assert isinstance(e, InvalidRoleError)
    else:
        raise AssertionError("expected InvalidRoleError")
    print("test_invalid_role_is_a_state_store_error: PASS")

def test_source_requirement_matches():
    exact = SourceRequirement(source_ref="docs/a.md@v1")
    assert exact.matches("docs/a.md@v1") and not exact.matches("docs/a.md@v2")

    prefix = SourceRequirement(source_ref_prefix="docs/")
    assert prefix.matches("docs/b.md@v1") and not prefix.matches("memory/x")

    both = SourceRequirement(source_ref="x", source_ref_prefix="docs/")
    assert both.matches("x") and both.matches("docs/z") and not both.matches("y")

    try:
        SourceRequirement()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a requirement with no matcher")
    print("test_source_requirement_matches: PASS")


def test_validator_applies_requirements_to_sources():
    class Doc(BaseModel):
        title: str

    validator = InputValidator()
    validator.add_requirement(SourceRequirement(source_ref="api/doc", required=True, schema=Doc))
    validator.add_requirement(SourceRequirement(source_ref_prefix="docs/", required=True))

    good = ContextSource(source_ref="api/doc", kind=SourceKind.RETRIEVED, content='{"title": "x"}')
    assert not validator.validate([good]).has_error          # raised TypeError before the fix

    drifted = ContextSource(source_ref="api/doc", kind=SourceKind.RETRIEVED, content='{"name": "x"}')
    assert any(i.check == "schema" for i in validator.validate([drifted]).errors())

    empty = ContextSource(source_ref="docs/policy.md", kind=SourceKind.RETRIEVED, content="   ")
    assert any(i.check == "completeness" for i in validator.validate([good, empty]).errors())
    print("test_validator_applies_requirements_to_sources: PASS")

def make_loop(model, *, max_turns=2, state=None, hooks=None):
    class SearchArgs(BaseModel):
        query: str

    async def search_docs(query: str) -> str:
        return f"results for {query}"

    registry = ToolRegistry()
    registry.register("search_docs", SearchArgs, description="search")(search_docs)
    context = ContextManager(token_budget=4000)
    context.set_system_prompt("sys")
    state = state or StateStore()
    loop = ExecutionLoop(
        model_client=model, tool_registry=registry, context_manager=context,
        state_store=state, hooks=hooks, max_turns=max_turns,
    )
    return loop, state

async def test_max_turns_is_per_run_on_reused_store():
    model = ScriptedModelClient([ModelResponse(content=f"a{i}") for i in range(3)])
    loop, state = make_loop(model, max_turns=2)
    for i in range(3):
        result = await loop.run(f"q{i}")
        assert result.end_reason == TurnEndReason.FINAL_ANSWER, f"run {i} ended with {result.end_reason}"
        assert result.turns_used == 1
    assert model.calls == 3          # the third run used to return MAX_TURNS with zero model calls
    assert state.turn_count == 3     # session counter stays cumulative
    print("test_max_turns_is_per_run_on_reused_store: PASS")


async def test_runaway_agent_is_capped_per_run_on_reused_store():
    class LoopingModel:
        def __init__(self): self.calls = 0
        async def complete(self, prompt, tool_schemas):
            self.calls += 1
            return ModelResponse(content="", tool_calls=[
                ToolCallRequest("search_docs", {"query": "x"}, f"c{self.calls}")])

    model = LoopingModel()
    loop, _ = make_loop(model, max_turns=2)
    for _ in range(2):
        result = await loop.run("again")
        assert result.end_reason == TurnEndReason.MAX_TURNS
    assert model.calls == 4          # 2 per run; the second run used to get 0
    print("test_runaway_agent_is_capped_per_run_on_reused_store: PASS")

def make_tracer(hooks):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))   # local provider, no global state touched
    return HarnessTracer(hooks, tracer=provider.get_tracer("test")), exporter


async def test_tracing_matches_tool_spans_by_call_id():
    hooks = LifecycleHooks()
    tracer, exporter = make_tracer(hooks)

    def call_ctx(tool, cid):
        return HookContext(HookEvent.ON_TOOL_CALL, "s1", 1, data={"tool_name": tool, "call_id": cid})

    await hooks.fire(HookEvent.ON_TOOL_CALL, call_ctx("search", "c1"))
    await hooks.fire(HookEvent.ON_TOOL_CALL, call_ctx("fetch", "c2"))

    ok = ToolResult(tool_name="fetch", output="x", duration_s=0.1, attempts=1, succeeded=True)
    bad = ToolResult(tool_name="search", output=None, duration_s=0.1, attempts=1, succeeded=False, error="boom")
    # results arrive out of order: c2 finishes before c1
    await hooks.fire(HookEvent.ON_TOOL_RESULT, HookContext(
        HookEvent.ON_TOOL_RESULT, "s1", 1, data={"tool_name": "fetch", "call_id": "c2", "result": ok}))
    await hooks.fire(HookEvent.ON_TOOL_RESULT, HookContext(
        HookEvent.ON_TOOL_RESULT, "s1", 1, data={"tool_name": "search", "call_id": "c1", "result": bad}))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["tool.fetch"].attributes["harborline.succeeded"] is True
    assert spans["tool.search"].attributes["harborline.succeeded"] is False   # old code swapped these
    assert tracer._open_spans == {}
    print("test_tracing_matches_tool_spans_by_call_id: PASS")


async def test_tracing_does_not_leak_on_duplicate_call_id():
    hooks = LifecycleHooks()
    tracer, exporter = make_tracer(hooks)
    ctx = HookContext(HookEvent.ON_TOOL_CALL, "s1", 1, data={"tool_name": "t", "call_id": "dup"})
    await hooks.fire(HookEvent.ON_TOOL_CALL, ctx)
    await hooks.fire(HookEvent.ON_TOOL_CALL, ctx)
    result = ToolResult(tool_name="t", output="x", duration_s=0.1, attempts=1, succeeded=True)
    await hooks.fire(HookEvent.ON_TOOL_RESULT, HookContext(
        HookEvent.ON_TOOL_RESULT, "s1", 1, data={"tool_name": "t", "call_id": "dup", "result": result}))
    assert len(exporter.get_finished_spans()) == 2
    assert tracer._open_spans == {}
    print("test_tracing_does_not_leak_on_duplicate_call_id: PASS")


async def test_tracing_closes_open_spans_when_model_call_fails():
    class ExplodingModel:
        async def complete(self, prompt, tool_schemas):
            raise RuntimeError("boom")

    hooks = LifecycleHooks()
    tracer, exporter = make_tracer(hooks)
    loop, _ = make_loop(ExplodingModel(), hooks=hooks)
    result = await loop.run("hi")
    assert result.end_reason == TurnEndReason.ERROR
    assert tracer._open_spans == {}                       # the llm.call span used to leak
    assert "llm.call" in [s.name for s in exporter.get_finished_spans()]
    print("test_tracing_closes_open_spans_when_model_call_fails: PASS")


async def main():
    await test_happy_path_with_tool_call()
    await test_validation_blocks_on_missing_required_source()
    await test_max_turns_exceeded_without_final_answer()
    await test_tool_execution_error_is_reported_not_raised()
    await test_tool_results_reach_second_prompt()
    test_history_budget_pins_latest_user_message()
    await test_registry_returns_errors_instead_of_raising()
    await test_bad_tool_calls_are_fed_back_to_model()
    test_state_store_round_trip()
    test_snapshot_restore_and_diff()
    test_get_memory_defaults()
    test_invalid_role_is_a_state_store_error()
    test_source_requirement_matches()
    test_validator_applies_requirements_to_sources()
    await test_max_turns_is_per_run_on_reused_store()
    await test_runaway_agent_is_capped_per_run_on_reused_store()
    await test_tracing_matches_tool_spans_by_call_id()
    await test_tracing_does_not_leak_on_duplicate_call_id()
    await test_tracing_closes_open_spans_when_model_call_fails()

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())