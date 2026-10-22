from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass , field , asdict
from enum import Enum
from typing import Any , Iterable


class StateStoreError(Exception):
    """Base class for exceptions in the state store."""

class InvalidRoleError(Exception):
    """Raised when a message is added  with an unrecognized role"""

class SnapshotNotFoundError(StateStoreError):
    """Raised when restoring a snapshot id th doesn't exist."""

class Role(str,Enum):
    SYSTEM="system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass(frozen=True)
class Message:
    role:Role
    content:str
    tool_name:str | None = None
    tool_call_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata : dict[str,Any] = field(default_factory=dict)

    def to_dict(self)->dict[str,Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d

@dataclass
class CostEntry:
    model:str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp:float = field(default_factory=time.time)

    @property
    def total_tokens(self)-> int:
        return self.prompt_tokens + self.completion_tokens

@dataclass
class CostSummary:
    total_prompt_tokens: int = 0
    total_completion_tokens:int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    by_model:dict[str,"CostSummary"] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

class StateStore:

    def __init__(self,session_id:str|None = None , max_turns:int | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.max_turns = max_turns
        self._lock = threading.RLock()
        self._messages: list[Message] = []
        self._memory: dict[str,Any] = {}
        self._cost_log:list[CostEntry] = []
        self._turn_count: int = 0
        self._snapshots:dict[str,dict[str,Any]] = {}
        self._created_at = time.time()

    def add_message(
        self,
        role:Role | str,
        content:str,
        *,
        tool_name:str | None = None,
        tool_call_id : str | None = None,
        metadata:dict[str,Any] | None = None,
    ) -> Message:
        """Append a message to history. Thread-safe."""
        if isinstance(role,str):
            try:
                role=Role(role)
            except ValueError as exc:
                raise InvalidRoleError(
                    f"Unknown role {role!r}. Expected one of "
                    f"{[r.value for r in Role]}."
                )from exc

        if role == Role.TOOL and not tool_name:
            raise StateStoreError("tool messages must set tool_name")

        message = Message(
            role = role,
            content=content,
            tool_name = tool_name,
            tool_call_id = tool_call_id,
            metadata = metadata or {}
        )
        with self._lock:
            self._messages.append(message)
        return message

    def get_history(self,*,limit:int | None = None) -> list[Message]:
        """Return a defensive copy of the message history, oldest first."""
        with self._lock:
            history = list(self._messages)
        if limit is not None:
            return history[-limit:]
        return history

    def clear_history(self)-> None:
        with self._lock:
            self._messages.clear()

    def increment_turn(self) -> int:
        with self._lock:
            self._turn_count +=1
            return self._turn_count

    @property
    def turn_count(self) -> int:
        with self._lock:
            return self._turn_count

    def turns_remaining(self) -> int | None :
        if self.max_turns is None:
            return None
        with self._lock:
            return max(0,self.max_turns-self._turn_count)

    def set_memory(self,key:str,value: Any) -> None:
        with self._lock:
            self._memory[key] = value

    def get_memory(self,key:str, default = Any) -> None:
        with self._lock:
            return self._memory.get(key,default)

    def delete_memory(self, key:str)-> None:
        with self._lock:
            self._memory.pop(key,None)

    def memory_keys(self)-> list[str]:
        with self._lock:
            return list(self._memory.keys())

    def record_cost(
        self,
        model:str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd:float,
    ) -> None:
        if prompt_tokens< 0 or completion_tokens < 0 or cost_usd<0:
            raise StateStoreError("cost fields must be non-negative")
        with self._lock:
            self._cost_log.append(
                CostEntry(model, prompt_tokens, completion_tokens, cost_usd)
            )

    def cost_summary(self) -> CostSummary:
        with self._lock:
            entries = list(self._cost_log)

        summary = CostSummary()
        for entry in entries:
            summary.total_prompt_tokens += entry.prompt_tokens
            summary.total_completion_tokens +=entry.completion_tokens
            summary.total_cost_usd += entry.cost_usd
            summary.call_count +=1

            per_model = summary.by_model.setdefault(entry.model,CostSummary())
            per_model.total_prompt_tokens += entry.prompt_tokens
            per_model.total_completion_tokens +=entry.completion_tokens
            per_model.total_cost_usd += entry.cost_usd
            per_model.call_count +=1

        return summary

    def exceeds_cost_budget(self,max_usd: float) -> bool:
        return self.cost_summary().total_cost_usd>=max_usd

    def snapshot(self,label:str| None = None) -> str:
        snapshot_id = label or str(uuid.uuid4())
        with self._lock:
            self._snapshots[snapshot_id] = {
                "messages":list(self._messages),
                "memory":dict(self._memory),
                "turn_count":self._turn_count,
                "cost_log":list(self._cost_log),
            }
            return snapshot_id
        
    def to_dict(self) -> dict[str,Any]:
        with self._lock:
            return {
                "session_id":self.session_id,
                "created_at":self._created_at,
                "turn_count":self._turn_count,
                "max_turns":self.max_turns,
                "messages":[m.to_dict() for m in self._messages],
                "memory": dict(self._memory),
                "cost_log": [asdict(c) for c in self._cost_log],
            }

    def to_json(self)-> str:
        return json.dumps(self.to_dict(),default=str)

    @classmethod
    def from_dict(cls , data:dict[str , Any])-> "StateStore":
        store = cls(
            session_id=data["session_id"],
            max_turns=data.get("max_turns", data.get("max_truns")),
        )
        store._created_at = data.get("created_at" , time.time())
        store._turn_count = data.get("turn_count",0)
        store._memory = dict(data.get("memory",{}))
        store._messages = [
            Message(
                role = Role(m["role"]),
                content = m["content"],
                tool_name = m.get("tool_name"),
                tool_call_id = m.get("tool_call_id"),
                timestamp = m.get("timestamp", time.time()),
                metadata = m.get("metadata",{}),
            )
            for m in data.get("messages",[])
        ]
        store._cost_log = [
            CostEntry(
                model = c["model"],
                prompt_tokens = c["prompt_tokens"],
                completion_tokens = c["completion_tokens"],
                cost_usd = c["cost_usd"],
                timestamp = c.get("timestamp",time.time())
            )
            for c in data.get("cost_log",[])
        ]
        return store

    @classmethod
    def from_json(cls,raw:str) -> "StateStore":
        return cls.from_dict(json.loads(raw))
