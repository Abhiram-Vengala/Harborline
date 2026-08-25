"""
eval/runner.py
 
evaluate() is the single entry point that ties the harness to the eval
layer: for each TestCase in a dataset, it spins up a fresh, isolated
harness session (new StateStore, new ContextManager), runs it, then hands
the output to every grader configured on that test case.
 
Isolation is deliberate: each test case gets its own StateStore rather
than sharing one across a dataset run, so case N's conversation history
can never leak into case N+1's context. A shared-state eval runner is a
classic source of "why did this test only fail when run after that other
one" flakiness.
 
Usage (local/dev):
 
    report = await evaluate(
        dataset=load_golden_dataset("eval/datasets/golden/rag_cases.yaml"),
        agent_factory=my_agent_factory,
        grader_registry=default_grader_registry(judge_client=my_judge),
    )
    print(report.summary())
 
`agent_factory()` must return a fresh (loop, context_manager, tool_registry)
tuple per call -- see AgentFactory below -- so each test case runs against
a clean instance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass , field
from pathlib import Path
from typing import Any , Awaitable , Callable

import yaml

from eval.metrics.cost_latency import CostBudgetGrader , LatencyBudgetGrader , RegressionGrader
from eval.metrics.deterministic import (
    ContainsGrader,
    ExactMatchGrader,
    RegexGrader,
    SimilarityGrader,
    ToolCallGrader,
)
from eval.types import GradeStatus , GradeResult , TestCase , Grader
from harness.core.execution_loop import ExecutionLoop , TurnEndReason
from harness.core.lifecycle_hooks import HookContext , HookEvent , LifecycleHooks

logger = logging.getLogger("harborline.eval.runner")

class DatasetError(Exception):
    pass

def load_golden_dataset(path: str| Path) -> list[TestCase]:
    """
    Load one YAML file of test cases, or every .yaml/.yml file in a
    directory (sorted by filename for deterministic ordering). Each file
    is a list of test-case dicts matching TestCase's fields.
    """

    path = Path(path)
    files = sorted(path.glob("*.y*ml")) if path.is_dir() else [path]
    if not files:
        raise DatasetError(f"no dataset files found at {path}")

    cases : list[TestCase] =[]
    seen_ids: set[str] = set()
    for file in files:
        raw = yaml.safe_load(file.read_text())
        if raw is None :
            continue
        if not isinstance(raw , list):
            raise DatasetError(f"{file}: expected a YAML list of test cases, got {type(raw).__name__}")
        for entry in raw :
            try:
                case = TestCase(**entry)
            except TypeError as e:
                raise DatasetError(f"{file}: malformed test case {entry.get('id', '?')!r}: {e}") from e

            if case.id in seen_ids:
                raise DatasetError(f"duplicate test case id {case.id!r} (in {file})")
            seen_ids.add(case.id)
            cases.append(case)
    return cases

GraderBuilder = Callable[[dict[str,Any]],Grader]

class GraderRegistry:
    """
    `test_case.graders` is a list of plain dicts (so datasets stay
    declarative YAML, not Python). This registry turns each dict's `type`
    field into a live Grader instance. Register custom grader types with
    `.register()` before calling evaluate() if a dataset uses a type not
    covered by default_grader_registry().
    """

    def __init__(self) -> None:
        self._builders: dict[str , GraderBuilder] = {}

    def register(self,type_name:str, builder: GraderBuilder) -> None:
        self._builders[type_name] = builder

    def build(self,config: dict[str , Any]) -> Grader:
        type_name = config.get("type")
        if type_name not in self._builders:
            raise DatasetError(
                f"unknown grader type {type_name!r}. Registered types: {sorted(self._builders)}"
            )
        params = {k: v for k , v in config.items() if k !="type"}
        return self._builders[type_name](params)

def default_grader_registry(* , judge_client: Any = None) -> GraderRegistry:
    """
    Registers every grader type in metrics/ that needs no extra runtime
    wiring beyond what's in the config dict. `judge_client`, if given,
    enables the "llm_judge" type -- omit it if a dataset never uses that
    type and you don't want to construct a judge client for a purely
    deterministic run.
    """
    registry = GraderRegistry()
    registry.register("exact_match", lambda p:ExactMatchGrader(**p))
    registry.register("similarity", lambda p: SimilarityGrader(**p))
    registry.register("contains", lambda p: ContainsGrader(**p))
    registry.register("regex", lambda p: RegexGrader(**p))
    registry.register("tool_calls", lambda p: ToolCallGrader(**p))
    registry.register("cost_budget", lambda p: CostBudgetGrader(**p))
    registry.register("latency_budget", lambda p: LatencyBudgetGrader(**p))
    registry.register("regression", lambda p: RegressionGrader(**p))

    if judge_client is not None :
        from eval.metrics.llm_judge import LLMJudgeGrader , Rubric

        def build_judge(p: dict[str,Any]) -> Grader:
            rubrics = [Rubric(**r) for r in p.pop("rubrics")]
            name = p.pop("name" , "llm_judge")
            return LLMJudgeGrader(name , judge_client=judge_client , rubrics=rubrics , **p)

        registry.register("llm_judge" , build_judge)
    return registry

@dataclass
class HarnessInstance:
    loop: ExecutionLoop
    hooks: LifecycleHooks

AgentFactory = Callable[[TestCase] , Awaitable[HarnessInstance]] | Callable[[TestCase] , HarnessInstance]

@dataclass
class CaseResult:
    test_case_id : str
    end_reason: TurnEndReason
    final_answer: str | None
    grades : list[GradeResult]
    cost_usd : float
    latency_s : float
    tools_called: list[str]

    @property
    def passed(self) -> bool:
        if not self.grades:
            return False
        return all(g.passed for g in self.grades)

    @property
    def has_grader_errors(self) -> bool:
        return any(g.status == GradeStatus.ERROR for g in self.grades)

@dataclass
class EvalReport:
    case_results: list[CaseResult] = field(default_factory=list)
    dataset_path: str | None = None

    @property 
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for c in self.case_results if c.passed) / len(self.case_results)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.case_results)

    @property
    def mean_latency_s(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(c.latency_s for c in self.case_results) / len(self.case_results)

    def failing_cases(self) -> list[CaseResult]:
        return [c for c in self.case_results if not c.passed]

    def cases_with_errors(self) -> list[CaseResult]:
        return [c for c in self.case_results if c.has_grader_errors]

    def summary(self) -> str:
        n = len(self.case_results)
        passed = sum( 1 for c in self.case_results if c.passed)
        return(
            f"{passed}/{n} passed ({self.pass_rate:.0%}), "
            f"total cost ${self.total_cost_usd:.4f}, "
            f"mean latency {self.mean_latency_s:.2f}s"
        )

    def to_dict(self) -> dict[str,Any]:
        return {
            "dataset_path": self.dataset_path,
            "pass_rate":self.pass_rate,
            "total_cost_usd": self.total_cost_usd,
            "mean_latency_s": self.mean_latency_s,
            "cases":[
                {
                    "id": c.test_case_id,
                    "passed": c.passed,
                    "end_reason": c.end_reason.value,
                    "cost_usd": c.cost_usd,
                    "latency_s": c.latency_s,
                    "tools_called": c.tools_called,
                    "grades": [
                        {"grader": g.grader_name, "status": g.status.value, "score": g.score, "reasoning": g.reasoning}
                        for g in c.grades
                    ],
                }
                for c in self.case_results
            ],
        }

async def _run_single_case(
    test_case : TestCase,
    agent_factory: AgentFactory,
    grader_registry: GraderRegistry,
    baselines : dict[str , dict[str,Any]] | None,
) -> CaseResult:
    instance = agent_factory(test_case)
    if hasattr(instance,"__await__"):
        instance = await instance

    tools_called : list[str] = []
    instance.hooks.register(
        HookEvent.ON_TOOL_CALL,
        lambda ctx : tools_called.append(ctx.data.get("tool_name", "?")),
    )

    start = time.monotonic()
    try:
        result = await instance.loop.run(test_case.input)
    except Exception:
        logger.exception("harness run raised for test case %r", test_case.id)
        return CaseResult(
            test_case_id=test_case.id, end_reason=TurnEndReason.ERROR, final_answer=None,
            grades=[GradeResult(grader_name="_run", status=GradeStatus.ERROR, score=0.0,
                                 reasoning="harness raised an unhandled exception during run()")],
            cost_usd=0.0, latency_s=time.monotonic() - start, tools_called=tools_called,
        )
    latency_s = time.monotonic() - start 
    cost_usd = instance.loop.state.cost_summary().total_cost_usd
    actual_output = result.final_answer or ""

    run_context: dict[str , Any] = {
        "tools_called": tools_called,
        "cost_usd": cost_usd,
        "latency_s": latency_s,
        "end_reason": result.end_reason,
        "context_text": "(context text not captured by default agent_factory)",
        "baseline": (baselines or {}).get(test_case.id),
    }
    grades: list[GradeResult] = []
    for grader_config in test_case.graders:
        grader = grader_registry.build(grader_config)
        try:
            grades.append(await grader.grade(actual_output=actual_output, test_case=test_case, run_context=run_context))
        except Exception as exc:  # noqa: BLE001 -- a grader crashing shouldn't crash the whole eval run
            logger.exception("grader %r raised while grading case %r", grader_config.get("type"), test_case.id)
            grades.append(GradeResult(
                grader_name=str(grader_config.get("type")), status=GradeStatus.ERROR, score=0.0,
                reasoning=f"grader raised {type(exc).__name__}: {exc}",
            ))
 
    return CaseResult(
        test_case_id=test_case.id,
        end_reason=result.end_reason,
        final_answer=result.final_answer,
        grades=grades,
        cost_usd=cost_usd,
        latency_s=latency_s,
        tools_called=tools_called,
    )
 
 
async def evaluate(
    dataset: list[TestCase],
    agent_factory: AgentFactory,
    grader_registry: GraderRegistry,
    *,
    baselines: dict[str, dict[str, Any]] | None = None,
    concurrency: int = 4,
    dataset_path: str | None = None,
) -> EvalReport:
    """
    Run every TestCase in `dataset` against a fresh agent instance from
    `agent_factory`, grade each result, and return an EvalReport.
 
    `concurrency` bounds how many cases run in parallel -- each case is
    fully isolated (own StateStore/ContextManager) so parallelism is safe,
    but real judge/model APIs have rate limits, hence a default of 4
    rather than "all at once."
    """
    import asyncio
 
    semaphore = asyncio.Semaphore(max(1, concurrency))
 
    async def bounded(case: TestCase) -> CaseResult:
        async with semaphore:
            return await _run_single_case(case, agent_factory, grader_registry, baselines)
 
    results = await asyncio.gather(*(bounded(case) for case in dataset))
    return EvalReport(case_results=list(results), dataset_path=dataset_path)
 