from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any , Protocol

from harness.runtime.failures import FailureKind
from harness.runtime.states import EpisodeState

class TerminationReason(str,Enum):
    SUCCESS_VERIFIED = "success_verified"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    COST_LIMIT = "cost_limit"
    RESOURCE_LIMIT = "resource_limit"
    POLICY_DENIED = "policy_denied"
    VERIFICATION_FAILED = "verification_failed"
    MODEL_FAILURE = "model_failure"
    TOOL_FAILURE = "tool_failure"
    USER_ABORT = "user_abort"
    SYSTEM_ABORT = "system_abort"
    LOOP_DETECTED = "loop_detected"

R = TerminationReason

FINAL_STATE: dict[TerminationReason,EpisodeState] = {
    R.SUCCESS_VERIFIED: EpisodeState.COMPLETED,
    R.MODEL_FAILURE: EpisodeState.FAILED,
    R.TOOL_FAILURE: EpisodeState.FAILED,
    R.VERIFICATION_FAILED: EpisodeState.FAILED,
    R.MAX_TURNS: EpisodeState.TERMINATED,
    R.TIMEOUT: EpisodeState.TERMINATED,
    R.COST_LIMIT: EpisodeState.TERMINATED,
    R.RESOURCE_LIMIT: EpisodeState.TERMINATED,
    R.POLICY_DENIED: EpisodeState.TERMINATED,
    R.USER_ABORT: EpisodeState.TERMINATED,
    R.SYSTEM_ABORT: EpisodeState.TERMINATED,
    R.LOOP_DETECTED: EpisodeState.TERMINATED,
}

REASON_FOR_FAILURE: dict[FailureKind , TerminationReason] = {
    FailureKind.MODEL_FAILURE: R.MODEL_FAILURE,
    FailureKind.TOOL_NOT_FOUND: R.MODEL_FAILURE,
    FailureKind.TOOL_ARGUMENT_FAILURE: R.MODEL_FAILURE,
    FailureKind.TOOL_FAILURE: R.TOOL_FAILURE,
    FailureKind.TOOL_TIMEOUT: R.TOOL_FAILURE,
    FailureKind.TOOL_OUTPUT_FAILURE: R.TOOL_FAILURE,
    FailureKind.POLICY_FAILURE: R.POLICY_DENIED,
    FailureKind.VERIFICATION_FAILURE: R.VERIFICATION_FAILED,
    FailureKind.LOOP_FAILURE : R.LOOP_DETECTED,
    FailureKind.RESOURCE_FAILURE: R.RESOURCE_LIMIT,
    FailureKind.CONTEXT_FAILURE: R.SYSTEM_ABORT,
    FailureKind.STATE_FAILURE: R.SYSTEM_ABORT,
}

@dataclass(frozen=True)
class Termination:
    reason: TerminationReason
    final_state:EpisodeState
    detail: str = ""

class _HasAbort(Protocol):
    abort_reason: TerminationReason | None
    abort_detail: str

class TerminationController:
    def __init__(self,tracker:Any):
        self.tracker = tracker

    @staticmethod
    def make(reason: TerminationReason , detail:str ="") -> Termination:
        return Termination(reason , FINAL_STATE[reason] ,detail)

    def success(self,detail: str = "verifier accepted the result") -> Termination:
        return self.make(R.SUCCESS_VERIFIED,detail)

    def for_failure(self, kind: FailureKind, detail: str = "") -> Termination:
        return self.make(REASON_FOR_FAILURE[kind],detail)

    def poll(self,episode: _HasAbort) -> Termination | None:
        """Called at every state boundary. Returns a Termination if the episode must stop now."""
        if episode.abort_reason is not None:
            return self.make(episode.abort_reason, episode.abort_detail or "abort requested")
        hit = self.tracker.exceeded()
        if hit is not None:
            reason , detail = hit
            return self.make(reason,detail)
        return None


    