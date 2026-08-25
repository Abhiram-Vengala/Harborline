from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass,field
from enum import Enum
from typing import Any, Awaitable , Callable

logger = logging.getLogger("harborline.lifecycle_hooks")

HookFunc = Callable[["HookContext"],Any] | Callable[["HookContext"],Awaitable[Any]]

class HookEvent(str,Enum):
    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"
    ON_TOOL_CALL = "on_tool_call"
    ON_TOOL_RESULT = "on_tool_result"
    ON_ERROR = "on_error"
    ON_MODEL_CALL = "on_model_call"
    ON_MODEL_RESPONSE = "on_model_response"

@dataclass
class HookContext:
    event:HookEvent
    session_id:str
    turn:int
    data:dict[str,Any] = field(default_factory=dict)
    timestamp:float = field(default_factory=time.time)

@dataclass
class _Resgistration:
    func:HookFunc
    critical:bool
    name:str

class HookExecutionError(Exception):
    """Raised only for critical hooks that failed and are allowed to propagate."""

    def __init__(self,hook_name:str , event:HookEvent,original:BaseException):
        self.hook_name = hook_name
        self.event = event
        self.original = original
        super().__init__(
            f"critical hook {hook_name!r} on {event.value} raised "
            f"{type(original).__name__}: {original}"
        )

class LifecycleHooks:
    def __init__(self) -> None:
        self._hooks : dict[HookEvent,list[_Resgistration]] = {e: [] for e in HookEvent}

    def register(
        self,
        event:HookEvent,
        func:HookFunc,
        *,
        critical: bool = False,
        name:str|None = None,
    ) -> None:
        reg = _Resgistration(func=func,critical=critical,name=name or getattr(func,"__name__","anonymous"))
        self._hooks[event].append(reg)

    def unregister_all(self,event:HookEvent|None = None)-> None:
        if event is None:
            self._hooks={e: [] for e in HookEvent}
        else:
            self._hooks[event] = []

    async def fire(self,event:HookEvent, ctx: HookContext) -> None:
        for reg in self._hooks[event]:
            try:
                result = reg.func(ctx)
                if inspect.isawaitable(result):
                    await result
            except Exception  as exc:
                if reg.critical:
                    raise HookExecutionError(reg.name, event,exc) from exc
                logger.warning(
                    "non-critical hook %r on %s raised %s: %s (isolated, turn continues)",
                    reg.name, event.value, type(exc).__name__, exc,
                )

    def registered(self, event:HookEvent) -> list[str]:
        return [r.name for r in self._hooks[event]]