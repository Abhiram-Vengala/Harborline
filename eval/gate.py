"""
eval/gate.py

The merge-blocking layer: takes an EvalReport (from eval/runner.py) and a
GatePolicy, decides pass/fail, and exits with a non-zero code on failure
so a CI workflow step fails the job. This is deliberately separate from
runner.py -- evaluate() answers "how did the agent do", gate.py answers
"is that good enough to ship", which are different questions with
different owners (an eng lead tunes gate thresholds without needing to
touch how evaluation itself works).

Two ways to use this:
  - as a library: `decision = check_gate(report, policy)`, inspect
    `decision.passed` / `decision.reasons` yourself
  - as a CLI, for a GitHub Actions step:
        python -m eval.gate --report report.json --min-pass-rate 0.9
    exits 0 on pass, 1 on fail, printing a human-readable summary either
    way so the failure is visible directly in the Actions log without
    needing to open an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from eval.runner import EvalReport


@dataclass
class GatePolicy:
    """
    Every threshold is optional (None = not checked) so a policy can be as
    strict or loose as the calling workflow needs. Defaults are
    deliberately conservative (min_pass_rate=1.0, i.e. every case must
    pass) -- loosen explicitly per-project rather than have a lenient
    default silently let regressions through.
    """

    min_pass_rate: float | None = 1.0
    max_grader_errors: int | None = 0  # a grader ERRORing (not FAILing) usually means a broken eval, not a broken agent
    max_total_cost_usd: float | None = None
    max_mean_latency_s: float | None = None
    # Case ids allowed to fail without failing the gate (e.g. a known-flaky
    # case pending investigation) -- use sparingly, this is a deliberate
    # escape hatch and shows up in the gate output every time it's used.
    allowed_failing_case_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str]
    report_summary: str

    def render(self) -> str:
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}: {self.report_summary}",
        ]
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        return "\n".join(lines)


def check_gate(report: EvalReport, policy: GatePolicy) -> GateDecision:
    reasons: list[str] = []

    failing = [c for c in report.failing_cases() if c.test_case_id not in policy.allowed_failing_case_ids]
    ignored = [c for c in report.failing_cases() if c.test_case_id in policy.allowed_failing_case_ids]

    if ignored:
        reasons.append(
            f"NOTE: {len(ignored)} failing case(s) allowed through by policy override: "
            f"{[c.test_case_id for c in ignored]}"
        )

    effective_total = len(report.case_results)
    effective_passed = effective_total - len(failing) - len(ignored) if effective_total else 0
    # Recompute pass rate treating allow-listed failures as passes for the
    # purpose of this check, but still report them above -- the override
    # is visible, not silent.
    effective_pass_rate = (
        (effective_total - len(failing)) / effective_total if effective_total else 0.0
    )

    if policy.min_pass_rate is not None and effective_pass_rate < policy.min_pass_rate:
        reasons.append(
            f"pass rate {effective_pass_rate:.1%} below required {policy.min_pass_rate:.1%} "
            f"({len(failing)} failing case(s): {[c.test_case_id for c in failing]})"
        )

    error_cases = report.cases_with_errors()
    if policy.max_grader_errors is not None and len(error_cases) > policy.max_grader_errors:
        reasons.append(
            f"{len(error_cases)} case(s) had grader ERRORs (max allowed {policy.max_grader_errors}): "
            f"{[c.test_case_id for c in error_cases]} -- these indicate a broken grader or judge "
            f"call, not necessarily a bad agent response, and should be fixed before they can be trusted"
        )

    if policy.max_total_cost_usd is not None and report.total_cost_usd > policy.max_total_cost_usd:
        reasons.append(
            f"total cost ${report.total_cost_usd:.4f} exceeds budget ${policy.max_total_cost_usd:.4f}"
        )

    if policy.max_mean_latency_s is not None and report.mean_latency_s > policy.max_mean_latency_s:
        reasons.append(
            f"mean latency {report.mean_latency_s:.2f}s exceeds budget {policy.max_mean_latency_s:.2f}s"
        )

    blocking_reasons = [r for r in reasons if not r.startswith("NOTE:")]
    return GateDecision(passed=not blocking_reasons, reasons=reasons, report_summary=report.summary())


# ---------------------------------------------------------------------- #
# CLI, for use as a GitHub Actions step
# ---------------------------------------------------------------------- #

def _load_report(path: Path) -> EvalReport:
    """
    Reconstructs just enough of an EvalReport from its serialized
    to_dict() JSON to run gate checks -- the CLI path doesn't need full
    CaseResult objects with live GradeResult instances, only the
    aggregates and per-case pass/fail facts check_gate() actually reads.
    """
    from eval.runner import CaseResult
    from eval.types import GradeResult, GradeStatus
    from harness.core.execution_loop import TurnEndReason

    raw = json.loads(path.read_text())
    cases = []
    for c in raw["cases"]:
        grades = [
            GradeResult(
                grader_name=g["grader"], status=GradeStatus(g["status"]),
                score=g["score"], reasoning=g.get("reasoning", ""),
            )
            for g in c["grades"]
        ]
        cases.append(CaseResult(
            test_case_id=c["id"],
            end_reason=TurnEndReason(c["end_reason"]),
            final_answer=None,
            grades=grades,
            cost_usd=c["cost_usd"],
            latency_s=c["latency_s"],
            tools_called=c.get("tools_called", []),
        ))
    return EvalReport(case_results=cases, dataset_path=raw.get("dataset_path"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a CI run against an eval report (see eval/runner.py EvalReport.to_dict()).")
    parser.add_argument("--report", required=True, type=Path, help="Path to a JSON eval report")
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-grader-errors", type=int, default=0)
    parser.add_argument("--max-total-cost-usd", type=float, default=None)
    parser.add_argument("--max-mean-latency-s", type=float, default=None)
    parser.add_argument("--allow-failing", action="append", default=[], help="Case id allowed to fail (repeatable)")
    args = parser.parse_args(argv)

    if not args.report.exists():
        print(f"FAIL: report file not found: {args.report}", file=sys.stderr)
        return 1

    report = _load_report(args.report)
    policy = GatePolicy(
        min_pass_rate=args.min_pass_rate,
        max_grader_errors=args.max_grader_errors,
        max_total_cost_usd=args.max_total_cost_usd,
        max_mean_latency_s=args.max_mean_latency_s,
        allowed_failing_case_ids=frozenset(args.allow_failing),
    )
    decision = check_gate(report, policy)
    print(decision.render())
    return 0 if decision.passed else 1


if __name__ == "__main__":
    sys.exit(main())