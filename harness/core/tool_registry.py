from __future__ import annotations

import asyncio
import  inspect
import logging
import time
from dataclasses import dataclass , field
from typing import Any , Awaitable , Callable

from pydantic import BaseModel , ValidationError

logger = logging.getLogger("harborline.tool_registry")

ToolFunc = Callable[...,Any] | Callable[...,Awaitable[Any]]

class ToolRegistryError(Exception):
    """Base class for tool registry failures"""

class ToolNotFoundError(ToolRegistryError):
    def __init__(self,name:str,available:list[str]):
        self.name = name
        self.available = available
        super().__init__(
            f"no tool registered as {name!r}. Available tools: {available}"
        )

class ToolAlreadyRegisteredError(ToolRegistryError):
    def __init__(self,name:str):
        super().__init__(f"tool {name!r} is already registered")

class ToolInputValidationError(ToolRegistryError):
    def __init__(self,name:str,errors:Any):
        self.name = name
        self.errors = errors
        super().__init__(f"invalid input for tool {name!r}: {errors}")

class ToolOutputValidationError(ToolRegistryError):
    def __init__(self,name:str , errors:Any):
        self.name = name
        self.errors = errors
        super().__init__(f"tool {name!r} returned output violating its schema: {errors}")

class ToolExecutionError(ToolRegistryError):
    """Wraps an exception raised inside a tool's own implementation."""
    def __init__(self,name:str , original:BaseException):
        self.name = name
        self.original = original
        super().__init__(f"tool {name!r} raised {type(original).__name__}: {original}")

class ToolTimeOutError(ToolRegistryError):
    def __init__(self,name:str , timeout_s : float):
        self.name = name
        self.timeout_s = timeout_s
        super().__init__(f"tool {name!r} exceeded timeout of {timeout_s}s")

@dataclass
class ToolResult:
    tool_name : str
    output:Any
    duration_s : float
    attempts : int
    succeeded : bool
    error: str | None = None
    error_type : str | None = None

@dataclass
class RegisteredTool:
    name : str
    description : str
    func:ToolFunc
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None = None
    timeout_s : float = 30.0
    retryable : bool = False
    max_retries : int = 2
    retry_backoff_s : float = 0.5
    is_async : bool = field(init=False, default=False)

    def __post_init__(self) -> None :
        self.is_async = inspect.iscoroutinefunction(self.func)

    def to_schema_dict(self) -> dict[str,Any]:
        """JSON-schema description, e.g. for exposing to an LLM's tool-use API."""
        return {
            "name": self.name,
            "description" : self.description,
            "parameters" : self.input_schema.model_json_schema(),
        }

def _describe_input_error(tool:RegisteredTool , exc: ValidationError) -> str:
    problems = ";".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<arguments>'}: {err['msg']}"
        for err in exc.errors()
    )
    return (
        f"invalid arguments for tool {tool.name!r}: {problems}. "
        f"Expected fields: {sorted(tool.input_schema.model_fields)}"
    )

class ToolRegistry:
    """
    Central registry of callable tools. The execution loop looks up tools
    here by name and calls `execute()`, never invoking a raw Python
    function directly, so every tool call goes through the same
    validation/timeout/retry path.
    """

    def __init__(self) -> None:
        self._tools: dict[str , RegisteredTool] = {}

    def register(
        self,
        name:str,
        input_schema:type[BaseModel],
        *,
        description:str ="",
        output_schema : type[BaseModel] | None = None ,
        timeout_s : float = 30.0,
        retryable: bool = False,
        max_retries: int = 2,
        retry_backoff_s : float = 0.5,
    ) -> Callable[[ToolFunc],ToolFunc]:
        """
        Decorator form:
 
            class SearchArgs(BaseModel):
                query: str
                top_k: int = 5
 
            @registry.register("search_docs", SearchArgs, description="...")
            def search_docs(query: str, top_k: int = 5) -> list[str]:
                ...
        """
        def decorator(func:ToolFunc)-> ToolFunc:
            if name in self._tools:
                raise ToolAlreadyRegisteredError(name)
            self._tools[name] = RegisteredTool(
                name = name ,
                description=description,
                func=func,
                input_schema=input_schema,
                output_schema=output_schema,
                timeout_s=timeout_s,
                retryable=retryable,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s
            )
            return func
        
        return decorator

    def unregister(self,name:str) -> None:
        self._tools.pop(name,None)

    def get(self,name:str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name , sorted(self._tools.keys())) from None

    def list_tools(self)-> list[RegisteredTool]:
        return list(self._tools.values())

    def schemas_for_llm(self) -> list[dict[str,Any]]:
        """Tool schemas formatted for handing to an LLM's tool-use / function-calling API."""
        return [t.to_schema_dict() for t in self._tools.values()]

    @staticmethod
    def _failure(name , start , error , error_type , attempts: int =0) -> ToolResult:
        return ToolResult(
            tool_name = name , output= None , duration_s=time.monotonic() -start,
            attempts= attempts , succeeded=False , error=error , error_type=error_type,
        )

    async def execute(self , name :str , raw_args:dict[str,Any]) -> ToolResult:
        """
        Validate `raw_args` against the tool's input schema, run it
        (respecting timeout/retry policy), validate the output if an
        output schema is set, and return a ToolResult. Never raises for
        tool-side failures -- those are captured in ToolResult.error so
        the execution loop can decide how to handle a failed tool call
        (e.g. feed the error back to the model) without a try/except at
        every call site. Registry-level programmer errors (unknown tool
        name, bad schema) still raise, since those indicate a bug in the
        calling code, not a runtime tool failure.
        """
        start = time.monotonic()

        try:
            tool = self.get(name)
        except ToolNotFoundError as e:
            return self._failure(name,start,str(e),"tool_not_found")
        try:
            validated = tool.input_schema.model_validate(raw_args)
        except ValidationError as e :
            return self._failure(
                name,
                start,
                _describe_input_error(tool, e),
                "invalid_arguments",
            )

        kwargs = validated.model_dump()
        attempts = 0
        max_attempts = tool.max_retries +1 if tool.retryable else 1
        start = time.monotonic()
        last_error : BaseException | None = None
        last_error_type = "execution_error"

        while attempts < max_attempts:
            attempts +=1 
            try:
                output = await self._run_with_timeout(tool , kwargs)
            except (ToolTimeOutError , ToolExecutionError) as e:
                last_error = e
                last_error_type = "timeout" if isinstance(e, ToolTimeOutError) else "execution_error"
                logger.warning("tool %r attempt %d/%d failed: %s", name, attempts, max_attempts, e)
                if attempts >= max_attempts:
                    break
                await asyncio.sleep(tool.retry_backoff_s*attempts)
                continue
            if tool.output_schema is not None:
                try:
                    output = tool.output_schema.model_validate(output)
                except ValidationError as e:
                    msg = str(ToolOutputValidationError(name , e.errors()))
                    return self._failure(name,start , msg , "invalid_output" , attempts)
            return ToolResult(
                tool_name = name , output=output , duration_s= time.monotonic()-start,
                attempts=attempts,succeeded=True,
            )
        return self._failure(
            name , start , str(last_error) if last_error else "unknown failure" , last_error_type , attempts
        )

    async def _run_with_timeout(self, tool: RegisteredTool, kwargs: dict[str , Any]) -> Any:
        try:
            if tool.is_async:
                return await asyncio.wait_for(tool.func(**kwargs),timeout=tool.timeout_s)
            # Run sync tools off the event loop so a slow/blocking tool
            # doesn't stall the whole harness.
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None , lambda: tool.func(**kwargs)),
                timeout=tool.timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise ToolTimeOutError(tool.name,tool.timeout_s) from e
        except Exception as e :
            raise ToolExecutionError(tool.name , e) from e
        