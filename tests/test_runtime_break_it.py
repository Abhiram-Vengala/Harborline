"""
Stage 1 break-it suite: run the real EpisodeRunner against models and tools
that misbehave, and check the episode ends for the RIGHT reason with the
failure attributed to the RIGHT subsystem.

Run: python tests/test_runtime_break_it.py   (or: python -m pytest tests/test_runtime_break_it.py)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from harness.core.execution_loop import ModelResponse, ToolCallRequest
from harness.core.tool_registry import ToolRegistry
from harness.runtime import (
    Budget, EpisodeConfig, EpisodeRunner, EpisodeState, FailureKind, PolicyDecision,
    Subsystem, TerminationReason, VerificationResult,
)

R = TerminationReason
K = FailureKind


# ---- fakes ---------------------------------------------------------------

class ScriptedModel:
    """Replays a script; after it runs out, repeats `fallback` (or raises)."""

    def __init__(self, script: list[Any], fallback: Any = "__raise__", on_call=None):
        self.script = list(script)
        self.fallback = fallback
        self.calls = 0
        self.on_call = on_call

    async def complete(self, prompt: str, tool_schemas: list[dict]) -> Any:
        self.calls += 1
        if self.on_call:
            self.on_call()
        item = self.script.pop(0) if self.script else self.fallback
        if item == "__raise__":
            raise RuntimeError("script exhausted")
        if isinstance(item, Exception):
            raise item
        return item


def call(name: str, args: dict | None = None, cid: str = "c") -> ModelResponse:
    return ModelResponse("", [ToolCallRequest(name, args or {}, cid)], 10, 5, 0.001, "fake")


def final(text: str = "done") -> ModelResponse:
    return ModelResponse(text, [], 10, 5, 0.001, "fake")


class Q(BaseModel):
    q: str = ""


class Report(BaseModel):
    count: int


def make_registry() -> tuple[ToolRegistry, dict]:
    reg = ToolRegistry()
    stats = {"search": 0}

    @reg.register("search", Q, description="search")
    async def search(q: str = "") -> str:
        stats["search"] += 1
        return f"results for {q}"

    @reg.register("hang", Q, description="never returns")
    async def hang(q: str = "") -> str:
        await asyncio.sleep(3600)
        return "unreachable"

    @reg.register("bad_report", Q, description="returns the wrong shape", output_schema=Report)
    async def bad_report(q: str = "") -> dict:
        return {"count": "not-a-number"}

    @reg.register("explode", Q, description="raises")
    async def explode(q: str = "") -> str:
        raise ValueError("kaboom")

    return reg, stats


def run(model, *, config=None, registry=None, **kw):
    reg, stats = (registry, {}) if registry else make_registry()
    runner = EpisodeRunner(model_client=model, tool_registry=reg, **kw)
    ep = asyncio.run(runner.run("do the task", config or EpisodeConfig()))
    return ep, stats


def kinds(ep):
    return [f.kind for f in ep.failures]


# ---- sanity ----------------------------------------------------------------

def test_happy_path_is_verified_success():
    ep, stats = run(ScriptedModel([call("search", {"q": "x"}), final("42")]))
    assert ep.termination.reason == R.SUCCESS_VERIFIED
    assert ep.state == EpisodeState.COMPLETED and ep.final_answer == "42"
    assert ep.failures == [] and stats["search"] == 1
    states = [t.to_state for t in ep.transitions]
    assert EpisodeState.POLICY_CHECK in states and EpisodeState.OBSERVATION in states
    assert [h.role for h in ep.history] == ["assistant", "tool", "assistant"]
    json.dumps(ep.to_dict())


# ---- 1. garbage model output ------------------------------------------------

def test_garbage_output_then_recovery():
    ep, _ = run(ScriptedModel([None, "just a string", ModelResponse(""), final("ok")]))
    assert ep.termination.reason == R.SUCCESS_VERIFIED
    assert kinds(ep) == [K.MODEL_FAILURE] * 3
    assert all(f.subsystem == Subsystem.MODEL for f in ep.failures)
    assert any(h.role == "runtime" for h in ep.history)  # the model is told what went wrong


def test_permanent_garbage_ends_model_failure():
    ep, _ = run(ScriptedModel([], fallback=None), config=EpisodeConfig(budget=Budget(max_retries=2)))
    assert ep.termination.reason == R.MODEL_FAILURE
    assert ep.state == EpisodeState.FAILED
    assert len(ep.failures) == 3  # initial + 2 retries
    assert "retry budget exhausted" in ep.termination.detail


def test_model_exception_and_malformed_tool_call_are_model_failures():
    bad_call = ModelResponse("", [ToolCallRequest("search", "not-a-dict", "c")])
    ep, _ = run(ScriptedModel([RuntimeError("api down"), bad_call, final()]))
    assert ep.termination.reason == R.SUCCESS_VERIFIED
    assert kinds(ep) == [K.MODEL_FAILURE, K.MODEL_FAILURE]
    assert "api down" in ep.failures[0].root_cause


# ---- 2. invalid tool --------------------------------------------------------

def test_invalid_tool_is_attributed_to_model_and_recoverable():
    ep, stats = run(ScriptedModel([call("rm_rf", {"path": "/"}), call("search", {"q": "a"}), final()]))
    assert ep.termination.reason == R.SUCCESS_VERIFIED
    f = ep.failures[0]
    assert f.kind == K.TOOL_NOT_FOUND and f.subsystem == Subsystem.MODEL
    assert f.action == "tool_call:rm_rf" and f.detected_in == EpisodeState.POLICY_CHECK
    assert stats["search"] == 1
    assert "does not exist" in [h for h in ep.history if h.role == "runtime"][0].content


def test_bad_arguments_attributed_to_model():
    bad = ModelResponse("", [ToolCallRequest("search", {"q": ["list", "not", "str"]}, "c")])
    ep, _ = run(ScriptedModel([bad, final()]))
    assert ep.failures[0].kind == K.TOOL_ARGUMENT_FAILURE
    assert ep.failures[0].subsystem == Subsystem.MODEL


def test_endless_invalid_tools_end_as_model_failure():
    model = ScriptedModel([], fallback=None)
    n = [0]

    def bump():
        n[0] += 1
        model.script.append(call(f"fake_{n[0]}", {}))  # different name each time

    model.on_call = bump
    ep, _ = run(model, config=EpisodeConfig(budget=Budget(max_retries=3)))
    assert ep.termination.reason == R.MODEL_FAILURE


# ---- 3. never terminates ----------------------------------------------------

def test_never_terminates_hits_max_turns():
    n = [0]

    class Endless:
        async def complete(self, prompt, schemas):
            n[0] += 1
            return call("search", {"q": f"query {n[0]}"})  # different args: not a loop

    ep, stats = run(Endless(), config=EpisodeConfig(budget=Budget(max_turns=4)))
    assert ep.termination.reason == R.MAX_TURNS and ep.state == EpisodeState.TERMINATED
    assert n[0] == 4 and stats["search"] == 4


def test_tool_call_budget_is_resource_limit():
    n = [0]

    class Endless:
        async def complete(self, prompt, schemas):
            n[0] += 1
            return call("search", {"q": str(n[0])})

    ep, stats = run(Endless(), config=EpisodeConfig(budget=Budget(max_tool_calls=2)))
    assert ep.termination.reason == R.RESOURCE_LIMIT
    assert stats["search"] == 2  # the 3rd call was stopped before executing


def test_cost_limit():
    expensive = ModelResponse("", [ToolCallRequest("search", {"q": "a"}, "c")], 10, 5, 0.50, "fake")
    ep, _ = run(ScriptedModel([expensive], fallback=expensive),
                config=EpisodeConfig(budget=Budget(max_cost_usd=0.60)))
    assert ep.termination.reason == R.COST_LIMIT


def test_token_limit():
    big = ModelResponse("", [ToolCallRequest("search", {"q": "a"}, "c")], 900, 200, 0.0, "fake")
    ep, _ = run(ScriptedModel([big], fallback=big), config=EpisodeConfig(budget=Budget(max_tokens=1000)))
    assert ep.termination.reason == R.RESOURCE_LIMIT


def test_wall_clock_timeout_with_fake_clock():
    now = [0.0]
    n = [0]

    class Slow:
        async def complete(self, prompt, schemas):
            n[0] += 1
            now[0] += 6.0
            return call("search", {"q": str(n[0])})

    reg, _ = make_registry()
    runner = EpisodeRunner(model_client=Slow(), tool_registry=reg, clock=lambda: now[0])
    ep = asyncio.run(runner.run("t", EpisodeConfig(budget=Budget(max_runtime_s=10))))
    assert ep.termination.reason == R.TIMEOUT and n[0] == 2


# ---- 4. repeated tool ---------------------------------------------------------

def test_repeated_identical_tool_call_is_loop_detected():
    same = call("search", {"q": "same"})
    ep, stats = run(ScriptedModel([], fallback=same))
    assert ep.termination.reason == R.LOOP_DETECTED and ep.state == EpisodeState.TERMINATED
    assert ep.failures[-1].kind == K.LOOP_FAILURE
    assert stats["search"] == 2  # the 3rd identical call never ran


# ---- 5. hung tool ---------------------------------------------------------------

def test_hung_tool_times_out_and_is_attributed_to_tool():
    cfg = EpisodeConfig(tool_timeout_s=0.05, budget=Budget(max_retries=1))
    t0 = time.monotonic()
    ep, _ = run(ScriptedModel([], fallback=call("hang", {"q": "x"})), config=cfg)
    assert time.monotonic() - t0 < 5
    assert kinds(ep) == [K.TOOL_TIMEOUT, K.TOOL_TIMEOUT]
    assert all(f.subsystem == Subsystem.TOOL for f in ep.failures)
    assert ep.termination.reason == R.TOOL_FAILURE and ep.state == EpisodeState.FAILED


def test_hung_model_times_out():
    class HungModel:
        async def complete(self, prompt, schemas):
            await asyncio.sleep(3600)

    ep, _ = run(HungModel(), config=EpisodeConfig(model_timeout_s=0.05, budget=Budget(max_retries=0)))
    assert ep.termination.reason == R.MODEL_FAILURE
    assert "exceeded" in ep.failures[0].root_cause


def test_tool_that_raises_is_tool_failure_then_recovers():
    ep, _ = run(ScriptedModel([call("explode", {"q": "a"}), call("search", {"q": "b"}), final()]))
    assert ep.failures[0].kind == K.TOOL_FAILURE and ep.failures[0].subsystem == Subsystem.TOOL
    assert ep.termination.reason == R.SUCCESS_VERIFIED


# ---- 6. malformed tool data -------------------------------------------------------

def test_tool_output_violating_schema_is_tool_output_failure():
    ep, _ = run(ScriptedModel([call("bad_report", {"q": "a"}), call("search", {"q": "b"}), final()]))
    assert ep.failures[0].kind == K.TOOL_OUTPUT_FAILURE
    assert ep.failures[0].subsystem == Subsystem.TOOL
    assert ep.termination.reason == R.SUCCESS_VERIFIED


def test_registry_returning_junk_object_is_tool_output_failure():
    class JunkRegistry:
        def list_tools(self):
            class T: name = "search"
            return [T()]
        def schemas_for_llm(self): return []
        async def execute(self, name, args): return {"not": "a ToolResult"}

    runner = EpisodeRunner(model_client=ScriptedModel([call("search", {"q": "a"})], fallback=final()),
                           tool_registry=JunkRegistry())
    ep = asyncio.run(runner.run("t", EpisodeConfig(budget=Budget(max_retries=0))))
    assert ep.failures[0].kind == K.TOOL_OUTPUT_FAILURE
    assert ep.termination.reason == R.TOOL_FAILURE


def test_huge_tool_output_is_truncated_not_fatal():
    reg = ToolRegistry()

    @reg.register("big", Q)
    async def big(q: str = "") -> str:
        return "x" * 100_000

    ep, _ = run(ScriptedModel([call("big"), final()]), registry=reg,
                config=EpisodeConfig(max_observation_chars=100))
    tool_item = [h for h in ep.history if h.role == "tool"][0]
    assert len(tool_item.content) < 200 and "truncated" in tool_item.content
    assert ep.termination.reason == R.SUCCESS_VERIFIED


# ---- 7. false success claim ----------------------------------------------------------

def did_the_work(ep, answer) -> VerificationResult:
    worked = any(h.role == "tool" and h.metadata.get("tool_name") == "search" for h in ep.history)
    return VerificationResult(worked, "" if worked else "no search was performed")


def test_false_success_is_never_verified():
    ep, _ = run(ScriptedModel([], fallback=final("All done, trust me!")), verifier=did_the_work,
                config=EpisodeConfig(budget=Budget(max_retries=2)))
    assert ep.termination.reason == R.VERIFICATION_FAILED
    assert ep.state == EpisodeState.FAILED and ep.final_answer is None
    assert kinds(ep) == [K.VERIFICATION_FAILURE] * 3
    assert "no search was performed" in ep.failures[0].root_cause


def test_false_claim_then_real_work_is_verified():
    ep, _ = run(ScriptedModel([final("done"), call("search", {"q": "a"}), final("really done")]),
                verifier=did_the_work)
    assert ep.termination.reason == R.SUCCESS_VERIFIED and ep.final_answer == "really done"
    assert kinds(ep) == [K.VERIFICATION_FAILURE]


def test_crashing_verifier_does_not_pass_anything():
    def boom(ep, answer): raise RuntimeError("verifier bug")

    ep, _ = run(ScriptedModel([final()]), verifier=boom)
    assert ep.termination.reason == R.VERIFICATION_FAILED
    assert ep.failures[0].subsystem == Subsystem.VERIFIER


# ---- policy, abort, safety net ----------------------------------------------------------

def test_fatal_policy_denial_ends_policy_denied():
    ep, stats = run(ScriptedModel([call("search", {"q": "secret"})]),
                    policy=lambda e, c: PolicyDecision(False, "no searching", fatal=True))
    assert ep.termination.reason == R.POLICY_DENIED and stats["search"] == 0


def test_non_fatal_policy_denial_is_fed_back_to_model():
    seen = []

    def policy(ep, c):
        seen.append(c.arguments["q"])
        return PolicyDecision(c.arguments["q"] != "bad", "bad query")

    ep, _ = run(ScriptedModel([call("search", {"q": "bad"}), call("search", {"q": "good"}), final()]), policy=policy)
    assert ep.termination.reason == R.SUCCESS_VERIFIED
    assert ep.failures[0].kind == K.POLICY_FAILURE and ep.failures[0].subsystem == Subsystem.POLICY


def test_abort_requested_before_run_is_honoured():
    from harness.runtime import Episode
    reg, _ = make_registry()
    model = ScriptedModel([final()])
    ep = Episode("t")
    ep.request_abort(R.USER_ABORT, "user pressed stop")
    asyncio.run(EpisodeRunner(model_client=model, tool_registry=reg).run_episode(ep))
    assert ep.termination.reason == R.USER_ABORT and model.calls == 0


def test_abort_mid_run_is_honoured():
    from harness.runtime import Episode
    reg, _ = make_registry()
    ep = Episode("t")
    n = [0]

    class Aborting:
        async def complete(self, prompt, schemas):
            n[0] += 1
            if n[0] == 2:
                ep.request_abort(R.USER_ABORT)
            return call("search", {"q": str(n[0])})

    asyncio.run(EpisodeRunner(model_client=Aborting(), tool_registry=reg).run_episode(ep))
    assert ep.termination.reason == R.USER_ABORT and n[0] == 2


def test_runtime_bug_is_caught_as_state_failure():
    def broken_policy(ep, c): raise RuntimeError("policy engine bug")

    ep, _ = run(ScriptedModel([call("search", {"q": "a"})]), policy=broken_policy)
    assert ep.termination.reason == R.SYSTEM_ABORT and ep.state == EpisodeState.TERMINATED
    assert ep.failures[-1].kind == K.STATE_FAILURE and ep.failures[-1].subsystem == Subsystem.RUNTIME


def test_context_builder_crash_is_context_failure():
    def broken(ep): raise ValueError("no context")

    ep, _ = run(ScriptedModel([final()]), context_builder=broken)
    assert ep.failures[0].kind == K.CONTEXT_FAILURE and ep.termination.reason == R.SYSTEM_ABORT


def test_retry_streak_resets_after_a_successful_tool_call():
    # fail, succeed, fail, succeed, fail, succeed, done: 3 failures total with max_retries=1
    script = []
    for i in range(3):
        script += [None, call("search", {"q": str(i)})]
    script.append(final())
    ep, _ = run(ScriptedModel(script), config=EpisodeConfig(budget=Budget(max_retries=1)))
    assert ep.termination.reason == R.SUCCESS_VERIFIED and len(ep.failures) == 3


def test_every_episode_ends_in_exactly_one_terminal_state_with_a_reason():
    scenarios = [
        ScriptedModel([final()]), ScriptedModel([], fallback=None),
        ScriptedModel([], fallback=call("search", {"q": "z"})), ScriptedModel([call("nope")], fallback=None),
    ]
    for m in scenarios:
        ep, _ = run(m, config=EpisodeConfig(budget=Budget(max_retries=1, max_turns=6)))
        assert ep.is_terminal and ep.termination is not None
        assert ep.state == ep.termination.final_state
        assert ep.explain()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                failed += 1; print(f"{name}: FAIL {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)