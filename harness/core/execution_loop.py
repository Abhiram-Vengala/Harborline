from __future__ import annotations

import logging
import time
from dataclasses import dataclass,field
from enum import Enum
from typing import Any, Protocol

from harness.core.context_manager import AssembledContext, ContextManager
from harness.core.lifecycle_hooks import HookEvent , HookContext , LifecycleHooks
from harness.core.state_store import Role , StateStore
from harness.core.tool_registry import ToolRegistry , ToolResult
from harness.validation.input_validator import InputValidator , Severity , ValidationResult

logger = logging.getLogger("Harborline.execution_loop")

class ExecutionLoopError(Exception):
    pass

class MaxTurnsExceededError(ExecutionLoopError):
    def __init__(self,max_turns:int):
        self.max_turns = max_turns
        super().__init__(f"exceeded max_turns={max_turns} without reaching a final answer")

class CostBudgetExceededError(ExecutionLoopError):
    def __init__(self,budget_usd:float , spent_usd:float):
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        super().__init__(f"cost budget of ${budget_usd:.4f} exceeded (spent ${spent_usd:.4f})")

class ContextValidationBlockedError(ExecutionLoopError):
    """
    Raised when input_validator finds ERROR-severity issues and the loop is
    configured to block rather than proceed. This is the concrete payoff of
    the input-validation layer: a bad context stops the turn before a model
    call is even made, instead of quietly producing a plausible-looking but
    ungrounded answer.
    """
    def __init__(self,result:ValidationResult):
        self.result = result
        super().__init__(f"context validation blocked this turn: {result.summary()}")

class ToolCallRequest:
    __slots__ = ("tool_name","arguments","call_id")

    def __init__(self,tool_name:str,arguments:dict[str,Any],call_id:str):
        self.tool_name = tool_name
        self.arguments = arguments
        self.call_id = call_id

@dataclass
class ModelResponse:
    content:str
    tool_calls : list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens : int=0
    completion_tokens : int =0
    cost_usd : float = 0.0
    model: str = "unknown"

    @property
    def is_final(self) -> bool:
        """A response with no tool calls is treated as the agent's final answer."""
        return len(self.tool_calls) == 0

class ModelClient(Protocol):
    async def complete(
        self,
        prompt:str,
        tool_schemas: list[dict[str,Any]],
    ) -> ModelResponse:...

class TurnEndReason(str,Enum):
    FINAL_ANSWERS= "final_answer"
    FINAL_ANSWER = "final_answer"
    MAX_TURNS = "max_turns"
    COST_BUDGET = "cost_budget"
    VALIDATION_BLOCKED = "validation_blocked"
    ERROR = "error"

@dataclass
class RunResult:
    final_answer: str | None
    end_reason: TurnEndReason
    turns_used: int
    state: StateStore
    validation_result : ValidationResult | None = None

class ExecutionLoop:
    def __init__(
        self,
        *,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        context_manager:ContextManager,
        state_store: StateStore,
        input_validator: InputValidator| None = None,
        hooks: LifecycleHooks | None = None,
        max_turns: int = 10,
        cost_budget_usd:float | None = None,
        block_on_validation_error: bool = True
    ):
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context = context_manager
        self.state = state_store
        self.validator = input_validator
        self.hooks = hooks or LifecycleHooks()
        self.max_turns = max_turns
        self.cost_budget_usd = cost_budget_usd
        self.block_on_validation_error = block_on_validation_error

    async def run(self, user_input: str) -> RunResult:
        """
        Run the agent to completion (or until a limit stops it) on a
        single user input. Conversation history from prior calls on the
        same StateStore is preserved, so this can be called repeatedly for
        a multi-turn chat session, or once per eval case with a fresh
        StateStore for isolated evaluation runs.
        """
        self.state.add_message(Role.USER, user_input)
 
        for _ in range(self.max_turns):
            turn = self.state.increment_turn()
 
            if self.max_turns and turn > self.max_turns:
                return self._finish(TurnEndReason.MAX_TURNS, turn)
 
            if self.cost_budget_usd is not None and self.state.exceeds_cost_budget(
                self.cost_budget_usd
            ):
                return self._finish(TurnEndReason.COST_BUDGET, turn)
 
            await self.hooks.fire(
                HookEvent.PRE_TURN,
                HookContext(HookEvent.PRE_TURN, self.state.session_id, turn),
            )
 
            assembled = self.context.build(history=self.state.get_history())
 
            validation_result = None
            if self.validator is not None:
                validation_result = self.validator.validate(assembled.sections)
                if validation_result.has_error and self.block_on_validation_error:
                    logger.error(
                        "turn %d blocked by input validation: %s",
                        turn, validation_result.summary(),
                    )
                    return self._finish(
                        TurnEndReason.VALIDATION_BLOCKED, turn, validation_result=validation_result
                    )
                if validation_result.issues:
                    logger.warning(
                        "turn %d: context validation found issues: %s",
                        turn, validation_result.summary(),
                    )
 
            try:
                response = await self._call_model(assembled, turn)
            except Exception as exc:  # noqa: BLE001 -- convert to ON_ERROR hook + structured end
                await self.hooks.fire(
                    HookEvent.ON_ERROR,
                    HookContext(
                        HookEvent.ON_ERROR, self.state.session_id, turn,
                        data={"error": str(exc), "stage": "model_call"},
                    ),
                )
                logger.exception("model call failed on turn %d", turn)
                return self._finish(TurnEndReason.ERROR, turn, validation_result=validation_result)
 
            self.state.record_cost(
                response.model, response.prompt_tokens, response.completion_tokens, response.cost_usd
            )
 
            if response.is_final:
                self.state.add_message(Role.ASSISTANT, response.content)
                await self.hooks.fire(
                    HookEvent.POST_TURN,
                    HookContext(
                        HookEvent.POST_TURN, self.state.session_id, turn,
                        data={"final": True, "content": response.content},
                    ),
                )
                return self._finish(
                    TurnEndReason.FINAL_ANSWERS, turn,
                    final_answer=response.content, validation_result=validation_result,
                )
 
            # Model asked for tool calls -- dispatch each, feed results back.
            self.state.add_message(
                Role.ASSISTANT, response.content,
                metadata={"tool_calls": [tc.tool_name for tc in response.tool_calls]},
            )
            for call in response.tool_calls:
                await self._dispatch_tool_call(call, turn)
 
            await self.hooks.fire(
                HookEvent.POST_TURN,
                HookContext(HookEvent.POST_TURN, self.state.session_id, turn, data={"final": False}),
            )
 
        return self._finish(TurnEndReason.MAX_TURNS, self.max_turns)

    async def _call_model(self,assembled: AssembledContext , turn:int) -> ModelResponse:
        prompt = self.context.render_prompt(assembled)
        await self.hooks.fire(
            HookEvent.ON_MODEL_CALL,
            HookContext(HookEvent.ON_MODEL_CALL,self.state.session_id,turn , data = {"prompt": prompt}),
        )
        response = await self.model_client.complete(prompt,self.tool_registry.schemas_for_llm())
        await self.hooks.fire(
            HookEvent.ON_MODEL_RESPONSE,
            HookContext(
                HookEvent.ON_MODEL_RESPONSE,self.state.session_id,turn,
                data={"response":response},
            ),
        )
        return response

    async def _dispatch_tool_call(self,call,turn:int) -> ToolResult:
        await self.hooks.fire(
            HookEvent.ON_TOOL_CALL,
            HookContext(
                HookEvent.ON_TOOL_CALL,self.state.session_id,turn,
                data = {"tool_name":call.tool_name,"arguments":call.arguments, "call_id":call.call_id},
            ),
        )
        result = await self.tool_registry.execute(call.tool_name,call.arguments)

        if result.succeeded:
            self.state.add_message(
                Role.TOOL,str(result.output),
                tool_name = call.tool_name, tool_call_id=call.call_id,
            )
        else:
            # Tool failure is reported back into the conversation as a
            # tool message so the model can see it and decide how to
            # recover (retry differently, ask the user, give up
            # gracefully) -- the loop doesn't silently swallow it, and it
            # doesn't crash the session either.
            self.state.add_message(
                Role.TOOL,f"ERROR : {result.error}",
                tool_name=call.tool_name , tool_call_id=call.call_id,
                metadata={"error":True},
            )
        await self.hooks.fire(
            HookEvent.ON_TOOL_RESULT,
            HookContext(
                HookEvent.ON_TOOL_RESULT, self.state.session_id, turn,
                data={"tool_name": call.tool_name, "result": result},
            ),
        )
        return result

    def _finish(
        self,
        reason: TurnEndReason,
        turns_used: int,
        *,
        final_answer: str | None = None,
        validation_result: ValidationResult | None = None,
    ) -> RunResult:
        return RunResult(
            final_answer=final_answer,
            end_reason=reason,
            turns_used=turns_used,
            state=self.state,
            validation_result=validation_result,
        ) 
