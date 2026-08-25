from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass , field
from enum import Enum
from typing import Any, Callable

DEFAULT_CHARS_PER_TOKEN = 4

class ContextManagerError(Exception):
    pass

class BudgetExceededError(ContextManagerError):
    def __init__(self, required: int , budget:int):
        self.required = required
        self.budget = budget
        super().__init__(
            f"assembled context needs ~{required} tokens, exceeds budget of {budget} "
            "even after dropping all droppable sources"
        )

class SourceKind(str,Enum):
    SYSTEM = "system"
    RETRIEVED = "retrieved"
    HISTORY = "history"
    MEMORY = "memory"

@dataclass
class ContextSource:
    source_ref :str
    kind: SourceKind
    content:str
    priority: int = 0
    droppable: bool = True
    fetched_at: float = field(default_factory=time.time)
    content_hash: str = field(default="")
    ttl_s: float|None = None 
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def is_expired(self,now : float| None = None) -> bool:
        if self.ttl_s is None:
            return False
        return (now or time.time())-self.fetched_at > self.ttl_s

@dataclass
class AssembledContext:
    system_prompt: str
    sections: list[ContextSource]
    estimated_tokens: int
    dropped_sources: list[str]

class ContextManager:
    def __init__(
        self , 
        token_budget: int = 8000 ,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        token_counter: Callable[[str],int] | None = None
    ):
        self.token_budget = token_budget
        self._chars_per_token = chars_per_token
        self._token_counter = token_counter or self._estimate_tokens
        self._sources: dict[str , ContextSource] = {}
        self._system_prompt: str = ""

    def _estimate_tokens(self , text:str) -> int :
        return max(1,len(text)//self._chars_per_token)

    def set_system_prompt(self,prompt:str) -> None:
        self._system_prompt = prompt

    def add_source(self,source:ContextSource) -> None:
        self._sources[source.source_ref] = source

    def remove_source(self,source_ref:str)-> None:
        self._sources.pop(source_ref,None)

    def get_source(self,source_ref:str) -> ContextSource | None:
        return self._sources.get(source_ref)

    def all_sources(self) -> list[ContextSource]:
        return list(self._sources.values())

    def expired_sources(self,now:float | None = None) -> list[ContextSource]:
        now = now or time.time()
        return [s for s in self._sources.values() if s.is_expired(now)]

    def clear(self) -> None:
        self._sources.clear()

    def build(self) -> AssembledContext:
        system_tokens = self._token_counter(self._system_prompt)
        mandatory = [s for s in self._sources.values() if not s.droppable]
        optional = [s for s in self._sources.values() if s.droppable]

        mandatory_tokens = sum(self._token_counter(s.content) for s in mandatory)
        used = system_tokens + mandatory_tokens
        if used> self.token_budget:
            raise BudgetExceededError(used,self.token_budget)

        optional.sort(key= lambda s:(s.priority, s.fetched_at),reverse=True)

        included: list[ContextSource] = list(mandatory)
        dropped: list[str] = []

        for source in optional:
            cost = self._token_counter(source.content)
            if used + cost <= self.token_budget:
                included.append(source)
                used+=cost
            else:
                dropped.append(source.source_ref)

        kind_order = {
            SourceKind.SYSTEM:0,
            SourceKind.RETRIEVED:1,
            SourceKind.MEMORY:2,
            SourceKind.HISTORY:3,
        }
        included.sort(key=lambda s:(kind_order.get(s.kind,99),-s.priority))

        return AssembledContext(
            system_prompt=self._system_prompt,
            sections=included,
            estimated_tokens=used,
            dropped_sources=dropped
        )

    def render_prompt(self,assembled: AssembledContext) -> str:
        parts = [assembled.system_prompt] if assembled.system_prompt else []
        for section in assembled.sections:
            if section.kind == SourceKind.SYSTEM:
                continue
            parts.append(f"[{section.kind.value}:{section.source_ref}]\n{section.content}")
        return "\n\n".join(parts)