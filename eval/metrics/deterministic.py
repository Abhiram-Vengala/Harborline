"""
eval/metrics/deterministic.py
 
Graders that need no model call: exact/fuzzy string match, regex, JSON
schema conformance, and a code grader that runs a caller-supplied Python
predicate against the output. These are the cheap, fast, zero-flake checks
that should make up the bulk of a PR-time gate -- reserve llm_judge for
what genuinely can't be checked deterministically (see runner.py's
recommended split between PR-time and nightly grader sets).
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any , Awaitable , Callable
from pydantic import BaseModel , ValidationError

from eval.types import GradeResult , GradeStatus , TestCase

class ExactMatchGrader :
    """Passes only if actual_output equals expected_output exactly (after optional normalization)."""

    def __init__(self , name : str = "exact_match" , * , case_sensitive: bool = True , strip: bool = True):
        self.name = name 
        self.case_sensitive = case_sensitive
        self.strip = strip

    def _normalize(self,text:str) -> str :
        if self.strip:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    async def grade(self, * , actual_output: str , test_case:TestCase , run_context: dict[str , Any]) -> GradeResult:
        if test_case.expected_output is None :
            return GradeResult(
                grader_name = self.name , status=GradeStatus.ERROR , score = 0.0 ,
                reasoning= "test case has no expected_output to compare against",
            )
        expected = self._normalize(test_case.expected_output)
        actual = self._normalize(actual_output)
        passed = expected == actual
        return GradeResult(
            grader_name= self.name,
            status= GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=1.0 if passed else 0.0,
            reasoning= "exact match" if passed else "output does not match expected exactly",
            details= {"expected":expected , "actual":actual},
        )

class SimilarityGrader:
    """
    Fuzzy match using sequence similarity ratio, for cases where wording
    can vary but meaning shouldn't (e.g. minor phrasing differences that
    exact_match would wrongly fail). Not a substitute for llm_judge on
    genuinely open-ended output -- this is for near-deterministic phrasing
    variance only.
    """
 
    def __init__(self, name: str = "similarity", *, threshold: float = 0.85):
        self.name = name
        self.threshold = threshold
 
    async def grade(self, *, actual_output: str, test_case: TestCase, run_context: dict[str, Any]) -> GradeResult:
        if test_case.expected_output is None:
            return GradeResult(
                grader_name=self.name, status=GradeStatus.ERROR, score=0.0,
                reasoning="test case has no expected_output to compare against",
            )
        ratio = difflib.SequenceMatcher(None, test_case.expected_output.strip(), actual_output.strip()).ratio()
        passed = ratio >= self.threshold
        return GradeResult(
            grader_name=self.name,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=ratio,
            reasoning=f"similarity {ratio:.3f} {'>=' if passed else '<'} threshold {self.threshold}",
            details={"ratio": ratio, "threshold": self.threshold},
        )
 
 
class ContainsGrader:
    """Passes if all of a set of required substrings appear in the output."""
 
    def __init__(self, name: str = "contains", *, must_contain: list[str] | None = None, case_sensitive: bool = False):
        self.name = name
        self.must_contain = must_contain or []
        self.case_sensitive = case_sensitive
 
    async def grade(self, *, actual_output: str, test_case: TestCase, run_context: dict[str, Any]) -> GradeResult:
        haystack = actual_output if self.case_sensitive else actual_output.lower()
        missing = []
        for needle in self.must_contain:
            n = needle if self.case_sensitive else needle.lower()
            if n not in haystack:
                missing.append(needle)
        passed = not missing
        score = 1.0 - (len(missing) / len(self.must_contain)) if self.must_contain else 1.0
        return GradeResult(
            grader_name=self.name,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=score,
            reasoning="all required substrings present" if passed else f"missing: {missing}",
            details={"missing": missing},
        )
 
 
class RegexGrader:
    def __init__(self, name: str = "regex", *, pattern: str, flags: int = 0):
        self.name = name
        self.pattern = pattern
        self._compiled = re.compile(pattern, flags)
 
    async def grade(self, *, actual_output: str, test_case: TestCase, run_context: dict[str, Any]) -> GradeResult:
        match = self._compiled.search(actual_output)
        passed = match is not None
        return GradeResult(
            grader_name=self.name,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=1.0 if passed else 0.0,
            reasoning=f"pattern {self.pattern!r} {'matched' if passed else 'did not match'}",
            details={"match": match.group(0) if match else None},
        )
 
 
class JSONSchemaGrader:
    """Passes if actual_output parses as JSON and validates against a pydantic schema."""
 
    def __init__(self, name: str = "json_schema", *, schema: type[BaseModel]):
        self.name = name
        self.schema = schema
 
    async def grade(self, *, actual_output: str, test_case: TestCase, run_context: dict[str, Any]) -> GradeResult:
        try:
            payload = json.loads(actual_output)
        except json.JSONDecodeError as exc:
            return GradeResult(
                grader_name=self.name, status=GradeStatus.FAIL, score=0.0,
                reasoning=f"output is not valid JSON: {exc}",
            )
        try:
            self.schema.model_validate(payload)
        except ValidationError as exc:
            return GradeResult(
                grader_name=self.name, status=GradeStatus.FAIL, score=0.0,
                reasoning="output JSON does not conform to schema",
                details={"errors": exc.errors()},
            )
        return GradeResult(grader_name=self.name, status=GradeStatus.PASS, score=1.0, reasoning="schema valid")

class ToolCallGrader:
    """
    Passes if the tools actually invoked during the run (passed in via
    run_context["tools_called"], populated by runner.py from the harness's
    ON_TOOL_CALL hook) match test_case.expected_tool_calls. Order-insensitive
    by default since most agents don't guarantee call order.
    """

    def __init__(self,name:str = "tool_calls" , * , order_sensitive: bool = False):
        self.name = name 
        self.order_sensitive = order_sensitive

    async def grade(self , *, actual_output:str , test_case:TestCase , run_context: dict[str , Any])-> GradeResult:
        actual_calls = run_context.get("tools_called" , [])
        expected = test_case.expected_tool_calls

        if self.order_sensitive:
            passed = actual_calls == expected
        else :
            passed = sorted(actual_calls) == sorted(expected)

        return GradeResult(
            grader_name= self.name,
            status = GradeStatus.PASS if passed else GradeStatus.FAIL,
            score = 1.0 if passed else 0.0,
            reasoning= "tool calls match expected" if passed else (
                f"expected {expected}, got {actual_calls}"
            ),
            details= {"expected":expected , "actual":actual_calls},
        )

class CodeGrader:
    """
    Runs a caller-supplied predicate against the output for checks that
    don't fit the other grader shapes (parsing a specific field out of a
    structured response, checking a numeric value falls in a range, etc).
    The predicate must be synchronous and side-effect free -- if you need
    async or I/O in a grader, write a dedicated Grader class instead of
    reaching for this escape hatch.
    """

    def __init__(self , name:str , predicate: Callable[[str , TestCase],bool],*,description: str = ""):
        self.name = name 
        self.predicate  = predicate
        self.description = description

    async def grade(self,*,actual_output: str , test_case: TestCase, run_context: dict[str,Any])-> GradeResult:
        try:
            passed = self.predicate(actual_output,test_case)
        except Exception as exc :
            return GradeResult(
                grader_name = self.name , status= GradeStatus.ERROR , score = 0.0 ,
                reasoning=f"predicate raised {type(exc).__name__}: {exc}",
            )
        return GradeResult(
            grader_name=  self.name,
            status= GradeStatus.PASS if passed else GradeStatus.FAIL,
            score = 1.0 if passed else 0.0,
            reasoning= self.description or ("predicate passed" if passed else "predicate failed"),
        )
    