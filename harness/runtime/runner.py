"""
EpisodeRunner drives an Episode through the state machine. It replaces the
bare `for` loop in ExecutionLoop for runs that need budgets, termination
reasons, failure attribution and loop detection.
 
model_client: anything with `async complete(prompt, tool_schemas) -> ModelResponse`
tool_registry: harness.core.tool_registry.ToolRegistry (or compatible)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass , field
from typing import Any , Callable

from harness.core.tool_registry import (
    ToolInputValidationError,
    ToolNotFoundError,
    ToolOutputValidationError,
)
from harness.runtime.budgets import BudgetTracker
from harness.runtime.episode import Episode , EpisodeConfig
from harness.runtime.failures import Failure , FailureKind , Subsystem
from harness.runtime.loop_detector import LoopDetector
from harness.runtime.states import EpisodeState
from harness.runtime.termination import Termination , TerminationController

logger = logging.getLogger("harborline.runtime.runner")
S = EpisodeState
K = FailureKind
SUB = Subsystem

@dataclass
class PolicyDecision:
    allowed: bool = True
    reason: str = ""
    fatal: bool = False

@dataclass
class VerificationResult:
    passed: bool 
    reason: str = ""

def allow_all(episode: Episode, call: Any) -> PolicyDecision:
    return PolicyDecision(True)

def accept_everything(episode: Episode , answer: str) -> VerificationResult:
    """Placeholder verifier. SUCCESS_VERIFIED is only as strong as the verifier you pass in."""
    return VerificationResult(True)

def default_context_builder(ep: Episode) -> str:
    lines = [f"TASK: {ep.task}"]
    for item in ep.history:
        lines.append(f"[{item.role}] {item.content}")
    return "\n".join(lines)

def validate_model_response(resp: Any) -> str | None :
    """Returns a description of what is wrong with a model response, or None if it is usable."""
    content = getattr(resp , "content" , None)
    calls = getattr(resp, "tool_calls",None)
    if content is None and calls is None :
        return f"not a model response ({type(resp).__name__})"
    if not isinstance(content,str):
        return "content is not a string"
    if not isinstance(calls,list):
        return "tool_call is not a list"
    for c in calls :
        name = getattr(c, "tool_name",None)
        if not isinstance(name,str) or not name.strip():
            return "tool call has no tool_name"
        if not isinstance(getattr(c,"arguments",None),dict):
            return f"tool call {name!r} arguments are not a dict"
    if not calls and not content.strip():
        return "empty response: no content and no tool calls"
    return None

@dataclass
class _Run:
    ep: Episode
    tracker: BudgetTracker
    ctl : TerminationController
    detector: LoopDetector
    prompt: str = ""
    pending: list[Any] = field(default_factory=list)
    current_call: Any = None
    observation: str = ""
    final_candidate: str = ""
    last_failure: Failure | None = None

async def _maybe_await(value : Any) -> Any:
    return await value if inspect.isawaitable(value) else value

class EpisodeRunner:
    def __init__(
        self,
        *,
        model_client: Any,
        tool_registry: Any,
        verifier: Callable[[Episode , str] , Any] = accept_everything,
        policy: Callable[[Episode,Any], Any] = allow_all,
        context_builder: Callable[[Episode],str] = default_context_builder,
        clock: Callable[[],float] = time.monotonic,
    ):
        self.model = model_client
        self.tool = tool_registry
        self.verifier = verifier
        self.policy = policy
        self.context_builder = context_builder
        self._clock = clock

    async def run(self, task:str , config:EpisodeConfig | None = None) -> Episode:
        return await self.run_episode(Episode(task,config))

    async def run_episode(self , ep: Episode) -> Episode:
        cfg = ep.config
        run = _Run(
            ep = ep,
            tracker=BudgetTracker(cfg.budget, self._clock),
            ctl=None,
            detector=LoopDetector(cfg.loop_threshold , window = max(10,cfg.loop_threshold)),
        )
        run.ctl = TerminationController(run.tracker)
        handlers = {
            S.CREATED: self._on_created,
            S.CONTEXT_BUILDING: self._on_context_building,
            S.MODEL_CALL: self._on_model_call,
            S.POLICY_CHECK: self._on_policy_check,
            S.TOOL_EXECUTION: self._on_tool_execution,
            S.OBSERVATION: self._on_observation,
            S.VERIFICATION: self._on_verification,
            S.RECOVERY: self._on_recovery,
        }
        while not ep.is_terminal:
            try:
                stop = run.ctl.poll(ep)
                if stop is not None:
                    ep.finish(stop)
                    continue
                await handlers[ep.state](run)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("runtime crashed in state %s" , ep.state.value)
                failure = ep.record_failure(
                    K.STATE_FAILURE , SUB.RUNTIME , f"state:{ep.state.value}",
                    f"unexpected {type(exc).__name__}: {exc}",
                )
                ep.finish(run.ctl.for_failure(K.STATE_FAILURE, failure.describe()))
        return ep

    def _recover(self, run: _Run, kind: FailureKind, subsystem: Subsystem, action: str,
                 root_cause: str, *, details: dict[str, Any] | None = None,
                 feedback: str | None = None) -> None:
        ep = run.ep
        failure = ep.record_failure(kind , subsystem , action , root_cause , details)
        run.last_failure = failure
        run.tracker.record_retry()
        ep.add_history("runtime",feedback or f"Your last action failed ({kind.value}): {root_cause}. Try a different approach.")
        ep.transition(S.RECOVERY , note= kind.value)

    def _fatal(self , run: _Run, kind: FailureKind , subsystem: Subsystem , action: str , root_cause: str)-> None:
        failure = run.ep.record_failure(kind , subsystem , action , root_cause)
        run.last_failure = failure
        run.ep.finish(run.ctl.for_failure(kind , failure.describe()))

    async def _on_created(self, run: _Run) -> None:
        run.ep.transition(S.CONTEXT_BUILDING , "episode started")

    async def _on_context_building(self , run: _Run) -> None:
        ep = run.ep
        run.tracker.begin_turn()
        ep.turn = run.tracker.turns
        stop = run.ctl.poll(ep)
        if stop is not None:
            ep.finish(stop)
            return
        try:
            run.prompt = self.context_builder(ep)
        except Exception as exc:
            self._fatal(run , K.CONTEXT_FAILURE , SUB.CONTEXT , "context_build",
                        f"context builder raised {type(exc).__name__}: {exc}")
            return
        ep.transitions(S.MODEL_CALL)

    async def _on_model_call(self , run: _Run) -> None:
        ep = run.ep
        try:
            resp = await asyncio.wait_for(
                self.model.complete(run.prompt , self.tool.schemas_for_llm()),
                ep.config.model_timeout_s,
            )
        except asyncio.TimeoutError:
            self._recover(run , K.MODEL_FAILURE , SUB.MODEL , "model_call" , f"model call exceeded {ep.config.model_timeout_s}s")
            return
        except Exception as exc:
            self._recover(run , K.MODEL_FAILURE, SUB.MODEL , "model_call" , f"model raised {type(exc).__name__}: {exc}")
            return
        problem = validate_model_response(resp)
        if problem:
            self._recover(run, K.MODEL_FAILURE, SUB.MODEL, "model_call",
                          f"malformed model response: {problem}",
                          details={"response_repr": repr(resp)[:200]})
            return

        run.tracker.record_usage(
            getattr(resp , "prompt_tokens" , 0) or 0 ,
            getattr(resp,"completion_tokens" , 0) or 0,
            getattr(resp ,"cost_usd", 0.0) or 0.0,
        )
        if resp.tool_calls:
            ep.add_history("assistant", resp.content, tool_calls=[c.tool_name for c in resp.tool_calls])
            run.pending = list(resp.tool_calls)
            ep.transitions(S.POLICY_CHECK)
        else:
            ep.add_history("assistant", resp.content)
            run.final_candidate = resp.content
            ep.transition(S.VERIFICATION)

    async def _on_policy_check(self , run: _Run) -> None:
        ep = run.ep
        call = run.pending.pop(0)
        run.current_call = call
        run.tracker.record_tool_call()
        stop = run.ctl.poll(ep)
        if stop is None:
            ep.finish(stop)
            return

        action = f"tool_call:{call.tool_name}"
        known = {t.name for t in self.tools.list_tools()}
        if call.tool_name not in known:
            run.pending.clear()
            self._recover(run, K.TOOL_NOT_FOUND, SUB.MODEL, action,f"model requested unknown tool {call.tool_name!r}",feedback=f"Tool {call.tool_name!r} does not exist. Available tools: {sorted(known)}")
            return

        decision = await _maybe_await(self.policy(ep,call))
        if not decision.allowed:
            run.pending.clear()
            reason = decision.reason or "denied by policy"
            if decision.fatal:
                self._fatal(run, K.POLICY_FAILURE, SUB.POLICY, action, reason)
            else:
                self._recover(run, K.POLICY_FAILURE, SUB.POLICY, action, reason,
                              feedback=f"Policy denied {call.tool_name}: {reason}")
                return

        if run.detector.record(call.tool_name, call.arguments):
            run.pending.clear()
            self._fatal(run, K.LOOP_FAILURE, SUB.MODEL, action,
                        f"{call.tool_name} called with identical arguments {run.detector.threshold} times")
            return
        ep.transitions(S.TOOL_EXECUTION)

    async def _on_tool_execution(self , run: _Run) -> None:
        ep = run.ep
        call = run.current_call
        action = f"tool_call:{call.tool_name}"
        try:
            result = await asyncio.wait_for(
                self.tools.execute(call.tool_name , call.arguments) , ep.config.tool_timeout_s
            )
        except asyncio.TimeoutError:
            run.pending.clear()
            self._recover(run, K.TOOL_TIMEOUT, SUB.TOOL, action,
                          f"tool did not answer within {ep.config.tool_timeout_s}s")
            return
        except ToolNotFoundError as exc:
            run.pending.clear()
            self._recover(run, K.TOOL_NOT_FOUND, SUB.MODEL, action, str(exc))
            return
        except ToolOutputValidationError as exc:
            run.pending.clear()
            self._recover(run, K.TOOL_OUTPUT_FAILURE, SUB.TOOL, action, str(exc))
            return
        except Exception as exc:
            run.pending.clear()
            self._recover(run, K.TOOL_FAILURE, SUB.TOOL, action,
                          f"registry raised {type(exc).__name__}: {exc}")
            return

        succeeded = getattr(result , "succeeded" , None)
        if not isinstance(succeeded , bool):
            run.pending.clear()
            self._recover(run, K.TOOL_OUTPUT_FAILURE, SUB.TOOL, action,
                          f"tool returned a malformed result object ({type(result).__name__})")
            return
        if not succeeded:
            run.pending.clear()
            err = str(getattr(result, "error" , None) or "tool failed")
            kind = K.TOOL_TIMEOUT if "timeout" in err.lower() else K.TOOL_FAILURE
            self._recover(run , kind , SUB.TOOL , action , err)
            return

        text = str(getattr(result , "output" , ""))
        limit = ep.config.max_observation_chars
        if len(text) > limit:
            text = text[: limit] + f"... [truncated {len(text) - limit} chars]"
        run.observation = text
        ep.transitions(S.OBSERVATION)

    async def _on_observation(self , run: _Run) -> None:
        ep = run.ep
        call = run.current_call
        ep.add_history("tool" , run.observation, tool_name=call.tool_name, call_id=getattr(call, "call_id", None))
        run.tracker.reset_retries()
        ep.transitions(S.POLICY_CHECK if run.pending else S.CONTEXT_BUILDING)

    async def _on_verification(self , run: _Run) -> None:
        ep = run.ep
        candidate = run.final_candidate
        try:
            result = await _maybe_await(self.verifier(ep , candidate))
            passed = getattr(result , "passed" , None)
            if not isinstance(passed , bool):
                raise TypeError(f"verifier returned {type(result).__name__}, expected VerificationResult")
        except Exception as exc :
            self._fatal(run , K.VERIFICATION_FAILURE , SUB.VERIFIER , "verification" , f"verifier crashed: {type(exc).__name__}: {exc}")
            return
        if passed:
            ep.final_answer = candidate
            ep.finish(run.ctl.success())
            return
        why = result.reason or "no reason given"
        self._recover(run, K.VERIFICATION_FAILURE, SUB.MODEL, "verification",
                      f"agent claimed completion but verifier rejected it: {why}",
                      feedback=f"Verifier rejected your answer: {why}. Do the work before finishing.")
        

    async def _on_recovery(self , run: _Run) -> None:
        ep = run.ep
        if run.tracker.retries_exhausted():
            f = run.last_failure
            assert f is not None
            detail = f"retry budget exhausted ({run.tracker.retries} consecutive failures); last: {f.describe()}"
            ep.finish(run.ctl.for_failure(f.kind , detail))
            return
        ep.transitions(S.CONTEXT_BUILDING, note ="retrying")