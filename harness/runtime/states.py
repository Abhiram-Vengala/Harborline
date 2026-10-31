"""
The rulebook for where an episode may be and where it may go next.
Every state change in the runtime goes through this table.
"""

from __future__ import annotations

from enum import Enum


class EpisodeState(str,Enum):
    CREATED = "created"
    CONTEXT_BUILDING = "context_building"
    MODEL_CALL = "model_call"
    POLICY_CHECK = "policy_check"
    TOOL_EXECUTION = "tool_execution"
    OBSERVATION = "observation"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"

S = EpisodeState

TRANSITIONS: dict[EpisodeState , frozenset[EpisodeState]] = {
    S.CREATED: frozenset({S.CONTEXT_BUILDING , S.TERMINATED}),
    S.CONTEXT_BUILDING: frozenset({S.MODEL_CALL , S.FAILED ,S.TERMINATED}),
    S.MODEL_CALL: frozenset({S.POLICY_CHECK , S.VERIFICATION , S.RECOVERY , S.FAILED , S.TERMINATED}),
    S.POLICY_CHECK: frozenset({S.TOOL_EXECUTION , S.RECOVERY , S.FAILED , S.TERMINATED}),
    S.TOOL_EXECUTION: frozenset({S.OBSERVATION , S.RECOVERY , S.FAILED , S.TERMINATED}),
    S.OBSERVATION: frozenset({S.POLICY_CHECK , S.CONTEXT_BUILDING , S.FAILED , S.TERMINATED}),
    S.VERIFICATION: frozenset({S.COMPLETED , S.RECOVERY , S.FAILED , S.TERMINATED}),
    S.RECOVERY: frozenset({S.CONTEXT_BUILDING , S.FAILED , S.TERMINATED}),
    S.COMPLETED: frozenset(),
    S.FAILED: frozenset(),
    S.TERMINATED: frozenset(),
}

TERMINAL_STATES = frozenset( s for s , nxt in TRANSITIONS.items() if not nxt)

class IllegalTransitionError(Exception):
    def __init__(self,current: EpisodeState , attempted: EpisodeState):
        self.current = current 
        self.attempted = attempted
        allowed = sorted(s.value for s in TRANSITIONS[current])
        super().__init__(
            f"illegal transition {current.value} -> {attempted.value}; "
            f"allowed from {current.value}: {allowed}"
        )

def is_legal(current: EpisodeState , nxt: EpisodeState) -> bool:
    return nxt in TRANSITIONS[current]