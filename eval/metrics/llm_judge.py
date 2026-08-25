"""
eval/metrics/llm_judge.py
 
G-Eval style grading: a judge model scores the agent's output against a
rubric, producing both a numeric score and a chain-of-thought rationale,
rather than a bare pass/fail. This is for the genuinely open-ended cases
deterministic.py can't cover -- correctness of a free-text explanation,
tone, whether an answer is grounded in the provided context, etc.
 
Two things production judge implementations tend to get wrong, that this
addresses directly:
 
  1. Trusting the judge's raw text output. Judges are asked to return
     strict JSON and the response is parsed defensively -- a judge that
     returns malformed output produces a GradeStatus.ERROR, never a false
     pass or a crash.
  2. Single-sample noise. LLM judges are not perfectly consistent between
     calls. `num_samples > 1` runs the judge multiple times and takes the
     mean score, which meaningfully reduces threshold-flakiness in a CI
     gate compared to a single judge call.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from typing import Any , Protocol

from eval.types import GradeResult , GradeStatus , TestCase

logger = logging.getLogger("harborline.eval.llm_judge")

class JudgeModelClient(Protocol):
    """
    Minimal interface the judge needs from a model client -- deliberately
    smaller than harness.core.execution_loop.ModelClient (no tool_schemas)
    since the judge never calls tools, it only scores text.
    """
 
    async def complete_text(self, prompt: str) -> str: ...

@dataclass 
class Rubric:
    """
    A named criterion to score against, plus how much it weighs into the
    final aggregate score when a judge is configured with multiple
    criteria. `threshold` is the per-criterion pass bar; the grader's
    overall PASS/FAIL is "all criteria at or above their threshold" by
    default -- see LLMJudgeGrader(require_all_criteria=...).
    """
    name : str 
    criterion : str 
    threshold : float = 0.7
    weight : float = 1.0

_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator scoring an AI agent's response.
 
## Input given to the agent
{input}
 
## Context available to the agent
{context}
 
## Agent's actual output
{output}
 
## Evaluation criterion: {criterion_name}
{criterion}
 
Think step by step about whether the output satisfies this criterion, then
respond with ONLY a JSON object in this exact shape, no other text, no
markdown fences:
 
{{"reasoning": "<your step-by-step reasoning, 2-4 sentences>", "score": <float between 0.0 and 1.0>}}
"""

class JudgeParseError(Exception):
    def __init__(self,raw_response: str , original: Exception):
        self.raw_response = raw_response
        self.original = original
        super().__init__(f"could not parse judge response as JSON: {original}. Raw: {raw_response[:200]!r}")


def _parse_judge_response(raw:str) -> tuple[float , str]:

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
        score = float(payload["score"])
        reasoning = str(payload.get("reasoning" , ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc :
        raise JudgeParseError(raw , exc) from exc

    if not (0.0 <= score <=1.0):
        raise JudgeParseError(raw, ValueError(f"score {score} out of [0.0, 1.0] range"))
    return score , reasoning

class LLMJudgeGrader:

    def __init__(
        self,
        name : str ,
        judge_client : JudgeModelClient,
        rubrics: list[Rubric],
        *,
        num_samples: int = 1,
        require_all_criteria: bool = True,
    ):
        self.name = name
        self.judge_client = judge_client
        self.rubrics = rubrics
        self.num_samples = max(1, num_samples)
        self.require_all_criteria = require_all_criteria

    async def grade(self , * , actual_output: str , test_case: TestCase , run_context: dict[str , Any])-> GradeResult:
        context_text = run_context.get("context_text", "(no context recorded)")
        per_criterion: dict[str , float] = {}
        per_criterion_reasoning: dict[str , str] = {}
        parse_errors: list[str] = []

        for rubric in self.rubrics:
            samples : list[float] = []
            reasonings: list[str] = []
            for _ in range(self.num_samples):
                prompt = _JUDGE_PROMPT_TEMPLATE.format(
                    input = test_case.input,
                    context = context_text,
                    output = actual_output ,
                    criterion_name = rubric.name ,
                    criterion = rubric.criterion,
                )
                try:
                    raw = await self.judge_client.complete_text(prompt)
                    score , reasoning = _parse_judge_response(raw)
                    samples.append(score)
                    reasonings.append(reasoning)
                except JudgeParseError as exc:
                    logger.warning("judge parse failure for criterion %r: %s", rubric.name, exc)
                    parse_errors.append(f"{rubric.name}: {exc}")
                except Exception as exc :
                    logger.warning("judge call failed for criterion %r: %s", rubric.name, exc)
                    parse_errors.append(f"{rubric.name}: judge call raised {type(exc).__name__}: {exc}")

            if samples:
                per_criterion[rubric.name] = statistics.mean(samples)
                per_criterion_reasoning[rubric.name] = reasonings[-1]

        if not per_criterion:
            return GradeResult(
                grader_name=self.name , status = GradeStatus.ERROR , score = 0.0,
                reasoning="judge failed to produce a usable score on every criterion",
                details= {"parse_errors": parse_errors},
            )

        total_weight = sum(r.weight for r in self.rubrics if r.name in per_criterion)
        weighted_score = sum(
            per_criterion[r.name] * r.weight for r in self.rubrics if r.name in per_criterion
        )/total_weight

        criteria_passed = {
            r.name: per_criterion[r.name] >= r.threshold
            for r in self.rubrics
            if r.name in per_criterion
        }

        if self.require_all_criteria:
            overall_pass = all(criteria_passed.values()) and not parse_errors
        else :
            overall_pass = weighted_score >= min(r.threshold for r in self.rubrics)

        status = GradeStatus.PASS if overall_pass else GradeStatus.FAIL
        if parse_errors and not per_criterion:
            status = GradeStatus.ERROR

        reasoning_summary = "; ".join(
            f"{name}: {score:.2f} ({'pass' if criteria_passed.get(name) else 'fail'})"
            for name, score in per_criterion.items()
        )

        return GradeResult(
            grader_name=self.name,
            status=status,
            score=weighted_score,
            reasoning=reasoning_summary,
            details={
                "per_criterion_score": per_criterion,
                "per_criterion_reasoning": per_criterion_reasoning,
                "criteria_passed": criteria_passed,
                "parse_errors": parse_errors,
                "num_samples": self.num_samples,
            },
        )