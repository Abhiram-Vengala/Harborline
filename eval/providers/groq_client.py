"""
eval/providers/groq_client.py

Adapts Groq's API (OpenAI-compatible chat completions, served on Groq's
LPU hardware) to the same two protocols anthropic_client.py implements:
harness.core.execution_loop.ModelClient and
eval.metrics.llm_judge.JudgeModelClient. Neither execution_loop.py nor
llm_judge.py know or care that this file exists -- swapping providers is
just constructing a different client and handing it to agent_factory /
default_grader_registry, nothing in harness/ or eval/ changes.

Why Groq for this project: the free tier requires no credit card and
covers every model Groq hosts, gated only by rate limits (default: 30
requests/min, 1,000 requests/day, per-model token-per-minute caps) --
see https://console.groq.com/docs/rate-limits. That's enough to develop
against and run a small golden dataset in CI, though it's explicitly not
meant for production traffic. Cost is $0 on the free tier; the pricing
table below is kept for the (optional) case of upgrading to the paid
Developer tier later, so cost-budget graders keep working without code
changes if that happens -- on free tier it will just always compute to
$0.00 and the CostBudgetGrader/RegressionGrader graders effectively become
no-ops, which is fine.

Rate limits matter enough here to handle directly: a 429 from Groq is
retried with backoff honoring the `retry-after` header Groq sends, rather
than failing the whole eval run on the first rate-limit hit -- a full
golden-dataset run can easily need more calls than fit in one minute on
the free tier's 30 RPM cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import groq

from harness.core.execution_loop import ModelResponse, ToolCallRequest

logger = logging.getLogger("harborline.providers.groq")

# USD per million tokens, Groq's paid (Developer tier) on-demand pricing.
# Irrelevant on the free tier (cost is always $0 there) -- kept so this
# adapter doesn't need edits if the project later upgrades off free tier.
_PRICING_PER_MILLION_TOKENS = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
}
_DEFAULT_PRICING = {"input": 0.0, "output": 0.0}  # free tier: no charge regardless of model

_MAX_RETRIES = 4
_BASE_BACKOFF_S = 2.0


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, *, free_tier: bool) -> float:
    if free_tier:
        return 0.0
    pricing = _PRICING_PER_MILLION_TOKENS.get(model, _DEFAULT_PRICING)
    return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]


async def _call_with_retry(fn, *args, **kwargs):
    """
    Groq's free tier has real rate limits (30 RPM by default, per-model
    TPM caps) that a full eval run can hit. Retries on 429 honoring
    Groq's `retry-after` header when present, falling back to exponential
    backoff otherwise. Any other error propagates immediately -- only
    rate limiting is worth retrying transparently here.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except groq.RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                header_value = response.headers.get("retry-after")
                if header_value is not None:
                    try:
                        retry_after = float(header_value)
                    except ValueError:
                        retry_after = None
            wait_s = retry_after if retry_after is not None else _BASE_BACKOFF_S * (2 ** attempt)
            logger.warning("Groq rate limit hit (attempt %d/%d), waiting %.1fs", attempt + 1, _MAX_RETRIES, wait_s)
            await asyncio.sleep(wait_s)
    raise RuntimeError("unreachable")  # loop always returns or raises above


class GroqModelClient:
    """
    Implements harness.core.execution_loop.ModelClient. Default model is
    openai/gpt-oss-120b. Override it with the GROQ_MODEL environment variable
    when a different model is needed.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        free_tier: bool = True,
    ):
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.max_tokens = max_tokens
        self.free_tier = free_tier
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key with no credit card at "
                "https://console.groq.com/keys"
            )
        self._client = groq.AsyncGroq(api_key=key)

    async def complete(self, prompt: str, tool_schemas: list[dict[str, Any]]) -> ModelResponse:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema["parameters"],
                },
            }
            for schema in tool_schemas
        ]

        response = await _call_with_retry(
            self._client.chat.completions.create,
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            tools=tools or groq.NOT_GIVEN,
        )

        message = response.choices[0].message
        tool_calls: list[ToolCallRequest] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                logger.warning("Groq returned non-JSON tool arguments for %r: %r", call.function.name, call.function.arguments)
                arguments = {}
            tool_calls.append(ToolCallRequest(call.function.name, arguments, call.id))

        usage = response.usage
        cost = _estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens, free_tier=self.free_tier)

        return ModelResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            model=self.model,
        )


class GroqJudgeClient:
    """
    Implements eval.metrics.llm_judge.JudgeModelClient. Defaults to a
    smaller/faster free-tier model than the agent under test
    (openai/gpt-oss-20b) since judging a rubric is a lighter task than
    running the agent itself. Override it with GROQ_JUDGE_MODEL when needed.
    """

    def __init__(self, model: str | None = None, *, api_key: str | None = None, free_tier: bool = True):
        self.model = model or os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
        self.free_tier = free_tier
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key with no credit card at "
                "https://console.groq.com/keys"
            )
        self._client = groq.AsyncGroq(api_key=key)

    async def complete_text(self, prompt: str) -> str:
        response = await _call_with_retry(
            self._client.chat.completions.create,
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""