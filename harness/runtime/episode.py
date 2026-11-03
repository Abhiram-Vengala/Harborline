"""
The Episode: one run of the agent as a first-class object. It owns the
state, the legal-transition check, the history, the failures, and the
final termination. The runner mutates it; nobody else should.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass , field
from typing import Any

from harness.runtime.budgets import Budget
from harness.runtime.failures import Failure , FailureKind , Subsystem
from harness.runtime.states import (TERMINAL_STATES , TRANSITIONS , EpisodeState , IllegalTransitionError)
from harness.runtime.termination import Termination , TerminationReason

@dataclass(frozen=True)
class EpisodeConfig:
    budget: Budget = field(default_factory=Budget)
    loop_threshold: int = 3
    model_timeout_s: float = 60.0
    tool_timeout_s: float = 30.0
    max_observation_chars: int = 8000

@dataclass(frozen=True)
class Transition:
    from_state : EpisodeState
    to_state : EpisodeState
    turn: int
    timestamp: float
    note: str = ""

@dataclass
class HistoryTeam:
    role: str
    content: str #"assistant"|"tool"|"runtime"
    turn: int
    metadata: dict[str,Any] = field(default_factory=dict)

class Episode:
    def __init__(self,task:str , config:EpisodeConfig | None = None , episode_id : str | None = None):
        self.episode_id = episode_id or uuid.uuid4().hex[:12]
        self.task = task
        self.config = config or EpisodeConfig()
        self.state = EpisodeState.CREATED
        self.turn = 0
        self.transitions: list[Transition] = []
        self.failures: list[Failure] = []
        self.history: list[HistoryTeam] = []
        self.final_answer: str | None = None
        self.termination: Termination | None = None
        self.created_at = time.time()
        self.abort_reason: TerminationReason | None = None
        self.abort_detail: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition(self, to:EpisodeState , note: str= "") -> None:
        if to not in TRANSITIONS[self.state]:
            raise IllegalTransitionError(self.state,to)
        self.transitions.append(Transition(self.state, to , self.turn , time.time(), note))
        self.state = to

    def add_history(self,role:str , content:str , **metadata: Any) -> None:
        self.history.append(HistoryTeam(role , content , self.turn,metadata))

    def record_failure(
        self,
        kind: FailureKind,
        subsystem : Subsystem,
        action: str ,
        root_cause: str ,
        details: dict[str , Any] | None = None
    ) -> Failure :
        failure = Failure(
            episode_id=self.episode_id,
            turn= self.turn,
            action = action,
            subsystem=subsystem,
            kind=kind,
            root_cause=root_cause,
            detected_in=self.state,
            details= details or {},
        )
        self.failures.append(failure)
        return failure

    def request_abort(self, reason: TerminationReason = TerminationReason.USER_ABORT , detail: str = "") -> None:
        if reason not in (TerminationReason.USER_ABORT , TerminationReason.SYSTEM_ABORT):
            raise ValueError("abort reason must be USER_ABORT or SYSTEM_ABORT")
        self.abort_reason = reason
        self.abort_detail = detail

    def finish(self , termination: Termination) -> None:
        """Move to the final state and record why. COMPLETED is only reachable from VERIFICATION."""
        self.transition(termination.final_state , note=termination.reason.value)
        self.termination = termination

    def explain(self) -> str:
        """Answers: why exactly did this agent stop?"""
        if self.termination is None :
            return f"episode {self.episode_id} has not ended (state={self.state.value})"
        t = self.termination
        lines = [f"episode {self.episode_id} ended {self.state.value}: {t.reason.value} ({t.detail})"]
        if self.failures and t.reason != TerminationReason.SUCCESS_VERIFIED:
            lines.append(f"last failure: {self.failures[-1].describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str , Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "state": self.state.value,
            "turn": self.turn,
            "final_answer": self.final_answer,
            "termination": None if self.termination is None else {
                "reason": self.termination.reason.value,
                "final_state": self.termination.final_state.value,
                "detail": self.termination.detail,
            },
            "transitions":[
                {"from": t.from_state.value, "to": t.to_state.value, "turn": t.turn,
                 "timestamp": t.timestamp, "note": t.note}
                for t in self.transitions
            ],
            "failures": [f.to_dict() for f in self.failures],
            "history": [
                {"role": h.role, "content": h.content, "turn": h.turn, "metadata": h.metadata}
                for h in self.history
            ]
        }
