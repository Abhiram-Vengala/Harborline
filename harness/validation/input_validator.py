from __future__ import annotations
 
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
 
from pydantic import BaseModel, ValidationError

from harness.core.context_manager import ContextSource

class Severity(str,Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationIssue:
    source_ref: str
    check:str
    severity: Severity
    message : str

@dataclass
class ValidationResult:
    issues : list[ValidationIssue] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def errors(self) -> list[ValidationIssue]:
        return [ i for i in self.issues if i.severity ==Severity.ERROR]

    def summary(self) -> str:
        if not self.issues:
            return "all sources valid"
        counts: dict[Severity,int] = {}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity,0)+1
        parts = [f"{v} {k.value}" for k, v in counts.items()]
        return ",".join(parts)

@dataclass
class SourceRequirement:
    """
    Declares expectations for a given source_ref (or a prefix match via
    `source_ref_prefix`), so the validator knows what "valid" means for it.
    """

    source_ref : str | None = None
    source_ref_prefix: str | None = None
    required:bool = False
    schema: type[BaseModel] | None = None

    freshness_check: Callable[[ContextSource],bool] | None = None

    def matches(self,source_ref:str) -> bool:
        if self.source_ref is None:
            return source_ref == self.source_ref
        if self.source_ref_prefix is None:
            return source_ref.startswith(self.source_ref_prefix)
        return False

class InputValidator:
    def __init__(self) -> None:
        self._requirements: list[SourceRequirement] = []

    def add_requirement(self,requirement:SourceRequirement)-> None:
        self._requirements.append(requirement)

    def _requirements_for(self,source_ref:str) -> list[SourceRequirement]:
        return [ r for r in self._requirements if r.matches(source_ref)]

    def validate(
        self,
        sources: list[ContextSource],
        *,
        now:float | None = None,
    ) -> ValidationResult:
        now = now or time.time()
        result = ValidationResult()
        seen_refs = {s.source_ref for s in sources}

        for req in self._requirements:
            if not req.required:
                continue
            target = req.source_ref
            if target is not None and target not in seen_refs:
                result.issues.append(
                    ValidationIssue(
                        source_ref=target,
                        check="completeness",
                        severity=Severity.ERROR,
                        message=f"required source {target!r} is missing from context",
                    )
                )
        for source in sources:
            reqs = self._requirements_for(source.source_ref)
            # Freshness: TTL-based expiry, always checked regardless of
            # whether a requirement is registered for this source.
            if source.is_expired(now):
                result.issues.append(
                    ValidationIssue(
                        source_ref= source.source_ref,
                        check="freshness",
                        severity=Severity.WARNING,
                        message=(
                            f"source fetched {now - source.fetched_at:.0f}s ago, "
                            f"exceeds ttl of {source.ttl_s}s"
                        ),
                    )
                )

            for req in reqs:
                if req.required and not source.content.strip():
                    result.issues.append(
                        ValidationIssue(
                            source_ref=source.source_ref,
                            check="completeness",
                            severity= Severity.ERROR,
                            message="required source is present but empty",
                        )
                    )

                if req.schema is not None:
                    result.issues.append(self._check_schema(source,req.schema))
                if req.freshness_check is not None and not req.freshness_check(source):
                    result.issues.append(
                        ValidationIssue(
                            source_ref=source.source_ref,
                            check="freshness",
                            severity=Severity.ERROR,
                            message="custom freshness check failed",
                        )
                    )

        result.issues = [ i for i in result.issues if i is not None]
        return result

    @staticmethod
    def _check_schema(source:ContextSource , schema: type[BaseModel]) -> ValidationIssue | None:
        import json

        try:
            payload = json.loads(source.content)
        except (json.JSONDecodeError,TypeError):
            return ValidationIssue(
                source_ref=source.source_ref,
                check="schema",
                severity=Severity.ERROR,
                message="source content is not valid JSON, cannot check against schema",
            )

        try:
            schema.model_validate(payload)
        except ValidationError as e:
            return ValidationIssue(
                source_ref= source.source_ref,
                check="schema",
                severity=Severity.ERROR,
                message=f"schema drift detected: {e.errors()}",
            )
        return None

    @staticmethod
    def content_has_drifted(source: ContextSource, current_content:str) -> bool:
        """
        Re-fetch check: caller pulls the latest content for a source_ref
        from its origin (a doc store, an API) and passes it here to see if
        it's diverged from what's cached in the source's content_hash.
        Useful for a scheduled staleness sweep, separate from the
        per-turn TTL check.
        """
        current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()[:16]
        return current_hash != source.content_hash