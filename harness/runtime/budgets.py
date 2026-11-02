from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from harness.runtime.termination import TerminationReason

@dataclass(frozen=True)
class Budget:
    max_turns: int = 30
    max_tool_calls: int = 100
    max_runtime_s: float = 300.0
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_retries : int = 5

    def __post_init__(self) -> None :
        for name in ("max_turns","max_tool_calls","max_runtime_s","max_retries"):
            if getattr(self,name) <0:
                raise ValueError(f"{name} must be >=0")
        for name in ("max_tokens","max_cost_usd"):
            v = getattr(self,name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0 or None")

class BudgetTracker:
    """Counts usage. 'Exceeded' always means strictly greater than the limit."""

    def __init__(self , budget: Budget, clock: Callable[[],float] = time.monotonic):
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self.turns = 0
        self.tool_calls= 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.retries = 0

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._start

    def begin_turn(self) -> None:
        self.turns +=1

    def record_tool_call(self) -> None:
        self.tool_calls +=1

    def record_usage(self,prompt_tokens:int , completion_tokens: int , cost_usd: float = 0.0) -> None :
        self.tokens +=max(0,int(prompt_tokens)) + max(0,int(completion_tokens))
        self.cost_usd += max(0.0 , float(cost_usd))

    def record_retry(self) -> None:
        self.retries +=1

    def reset_retries(self) -> None:
        self.retries = 0

    def retries_exhausted(self) -> bool:
        return self.retries > self.budget.max_retries

    def exceeded(self) -> tuple[TerminationReason, str] | None:
        b = self.budget
        if self.elapsed_s > b.max_runtime_s:
            return TerminationReason.TIMEOUT , f"ran {self.elapsed_s:.1f}s, limit {b.max_runtime_s}s"

        if b.max_cost_usd is not None and self.cost_usd > b.max_cost_usd:
            return TerminationReason.COST_LIMIT , f"spent ${self.cost_usd:.4f}, limit ${b.max_cost_usd:.4f}"

        if b.max_tokens is not None and self.tokens > b.max_tokens:
            return TerminationReason.RESOURCE_LIMIT , f"used {self.tokens} tokens, limit {b.max_tokens}"

        if self.tool_calls > b.max_tool_calls:
            return TerminationReason.RESOURCE_LIMIT , f"{self.tool_calls} tool calls, limit {b.max_tool_calls}"

        if self.turns > b.max_turns:
            return TerminationReason.MAX_TURNS , f"turn {self.turns}, limit {b.max_turns}"
        return None
    