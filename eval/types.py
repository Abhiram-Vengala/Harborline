from __future__ import annotations
from dataclasses import dataclass , field
from enum import Enum
from typing import Any , Protocol , runtime_checkable

class GradeStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"

@dataclass
class GradeResult:
    grader_name:str
    status: GradeStatus
    score:float
    reasoning: str = ""
    details: dict[str , Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == GradeStatus.PASS

@runtime_checkable
class Grader(Protocol):
    """
    Every grader implements this. `name` is used in reports and in the
    gate's per-grader threshold config, so it must be stable across runs
    (don't derive it from anything that changes between calls).
    """
    name : str

    async def grade(self,*,actual_output: str , test_case: "TestCase" , run_context:dict[str,Any])-> GradeResult:...

@dataclass
class TestCase:
    """
    One golden test case. `id` must be stable and unique within a dataset
    -- it's the key runner.py and gate.py use to diff results across runs
    (e.g. "this case started failing after commit X").
    """

    id: str
    input : str
    expected_output: str | None = None
    expected_tool_calls: list[str] = field(default_factory=list)
    context_snapshot: str | None = None
    graders: list[dict[str,Any]] = field(default_factory=list)
    tags:list[str] = field(default_factory=list)
    max_cost_usd : float | None = None
    max_latency_s : float | None = None
    metadata: dict[str , Any] = field(default_factory=dict)
    
