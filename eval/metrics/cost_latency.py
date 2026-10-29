"""
eval/metrics/cost_latency.py
 
Graders for the two dimensions correctness-focused graders ignore entirely:
what a run cost and how long it took. Left unchecked, an agent can pass
every correctness check while quietly regressing into a much more
expensive or slower version of itself -- these graders catch that class of
regression explicitly, which is exactly the kind of check that belongs in
a CI gate rather than a human noticing a bill later.
 
Two graders:
  - CostBudgetGrader:    fails if a run exceeds an absolute cost ceiling
  - LatencyBudgetGrader: fails if a run exceeds an absolute latency ceiling
  - RegressionGrader:    fails if cost or latency grew more than a
                          tolerance versus a supplied baseline (for
                          catching gradual drift, not just hard breaches)
"""

from __future__ import annotations

from typing import Any 
from eval.types import GradeResult , GradeStatus , TestCase

class CostBudgetGrader:

    def __init__(self , name : str = "cost_budget" , * , default_max_usd: float | None = None):
        self.name = name
        self.default_max_usd = default_max_usd

    async def grade(self , * , actual_output: str , test_case:TestCase , run_context:dict[str,Any]) -> GradeResult:
        budget = test_case.max_cost_usd if test_case.max_cost_usd is not None else self.default_max_usd
        actual_cost = run_context.get("cost_usd")

        if budget is None:
            return GradeResult(
                grader_name= self.name , status=GradeStatus.ERROR , score = 0.0,
                reasoning="no cost budget configured (neither test case nor grader default)",
            )
        if actual_cost is None :
            return GradeResult(
                grader_name= self.name , status=GradeStatus.ERROR , score = 0.0,
                reasoning="run_context missing cost_usd -- runner did not record cost",
            )

        passed = actual_cost <= budget
        score = 1.0 if passed else max(0.0 , 1.0 - (actual_cost-budget)/budget)
        return GradeResult(
            grader_name= self.name,
            status= GradeStatus.PASS if passed else GradeStatus.FAIL,
            score = score ,
            reasoning=f"cost ${actual_cost:.5f} {'<=' if passed else '>'} budget ${budget:.5f}",
            details= {"actual_cost":actual_cost , "budget_usd":budget},
        ) 

class LatencyBudgetGrader:
    def __init__(self , name:str = "latency_budget" , * , default_max_s : float | None = None):
        self.name = name
        self.default_max_s = default_max_s

    async def grade(self , * , actual_output:str , test_case:TestCase , run_context:dict[str,Any]) -> GradeResult:
        budget = test_case.max_latency_s if test_case.max_latency_s is not None else self.default_max_s
        actual_latency = run_context.get("latency_s")
 
        if budget is None:
            return GradeResult(
                grader_name=self.name, status=GradeStatus.ERROR, score=0.0,
                reasoning="no latency budget configured (neither test case nor grader default)",
            )
        if actual_latency is None:
            return GradeResult(
                grader_name=self.name, status=GradeStatus.ERROR, score=0.0,
                reasoning="run_context missing latency_s -- runner did not record timing",
            )
 
        passed = actual_latency <= budget
        score = 1.0 if passed else max(0.0, 1.0 - (actual_latency - budget) / budget)
        return GradeResult(
            grader_name=self.name,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=score,
            reasoning=f"latency {actual_latency:.2f}s {'<=' if passed else '>'} budget {budget:.2f}s",
            details={"actual_s": actual_latency, "budget_s": budget},
        )

class RegressionGrader:
    """
    Compares this run's cost/latency against a per-test-case baseline
    (e.g. the last known-good run on main), failing if either metric grew
    by more than `tolerance` (fractional, 0.2 = 20% regression allowed).
    Baselines are supplied externally via run_context["baseline"] --
    runner.py is expected to load these from a prior report (see
    runner.py's `baseline_report` argument), this grader has no persistence
    of its own by design, so it stays a pure function of its inputs.
    """

    def __init__(self , name:str = "cost_latency_regression" , * , tolerance: float = 0.2):
        self.name = name
        self.tolerance = tolerance

    async def grade(self , * , actual_output: str , test_case : TestCase, run_context : dict[str,Any]) -> GradeResult:
        baseline = run_context.get("baseline")
        if not baseline:
            return GradeResult(
                grader_name= self.name , status=GradeStatus.ERROR , score=0.0,
                reasoning="no baseline available for this test case, skip or seed one first",
            )

        actual_cost = run_context.get("cost_usd")
        actual_latency = run_context.get("latency_s")
        baseline_cost = baseline.get("cost_usd")
        baseline_latency = baseline.get("latency_s")

        issues = []
        if actual_cost is not None and baseline_cost:
            cost_growth = (actual_cost-baseline_cost) / baseline_cost
            if cost_growth > self.tolerance:
                issues.append(f"cost grew {cost_growth:.0%} (tolerance {self.tolerance:.0%})")
        if actual_latency is not None and baseline_latency:
            latency_growth = (actual_latency-baseline_latency)/baseline_latency
            if latency_growth > self.tolerance:
                issues.append(f"latency grew {latency_growth:.0%} (tolerance {self.tolerance:.0%})")

        passed = not issues
        return GradeResult(
            grader_name=self.name ,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score= 1.0 if passed else 0.0,
            reasoning="within tolerance of baseline" if passed else "; ".join(issues),
            details={
                "actual_cost_usd": actual_cost, "baseline_cost_usd": baseline_cost,
                "actual_latency_s": actual_latency, "baseline_latency_s": baseline_latency,
            },
        )