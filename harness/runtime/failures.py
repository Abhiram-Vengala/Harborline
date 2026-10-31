"""
Failure taxonomy. A Failure says what broke (kind), where the fault
originated (subsystem), where the runtime noticed it (detected_in), and
ties it to an episode, turn and action.
"""

from __future__ import annotations

import time
from dataclasses import asdict , dataclass , field
from enum import Enum
from typing import Any

from harness.runtime.states import EpisodeState

class FailureKind(str,Enum):
    MODEL_FAILURE = "model_failure"
    CONTEXT_FAILURE = "context_failure"
    TOOL_FAILURE = "tool_failure"
    TOOL_ARGUMENT_FAILURE = "tool_argument_failure"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_OUTPUT_FAILURE = "tool_output_failure"
    POLICY_FAILURE = "policy_failure"
    LOOP_FAILURE = "loop_failure"
    RESOURCE_FAILURE = "resource_failure"
    VERIFICATION_FAILURE = "verification_failure"
    STATE_FAILURE = "state_failure"

class Subsystem(str,Enum):
    MODEL = "model"
    CONTEXT = "context"
    POLICY = "policy"
    TOOL = "tool"
    VERIFIER = "verifier"
    RUNTIME = "runtime"

@dataclass(frozen=True)
class Failure:
    episode_id : str
    turn: int
    action: str
    subsystem: Subsystem
    kind: FailureKind
    root_cause: str
    detected_in : EpisodeState
    timestamp: float = field(default_factory=time.time)
    details: dict[str,Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"turn {self.turn} [{self.action}] {self.kind.value} "
            f"(subsystem={self.subsystem.value}): {self.root_cause}"
        )

    def to_dict(self) -> dict[str ,Any]:
        d = asdict(self)
        d["subsystem"] = self.subsystem.value
        d["kind"] = self.kind.value
        d["detected_in"] = self.detected_in.value
        return d