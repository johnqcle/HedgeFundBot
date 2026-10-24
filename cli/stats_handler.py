import json
import threading
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._pending: dict[Any, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter and note the node and model of this call."""
        metadata = kwargs.get("metadata") or {}
        params = kwargs.get("invocation_params") or {}
        with self._lock:
            self.llm_calls += 1
            self._pending[kwargs.get("run_id")] = {
                "node": metadata.get("langgraph_node") or "?",
                "model": params.get("model") or params.get("model_name") or metadata.get("ls_model_name") or "?",
            }

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        with self._lock:
            call = self._pending.pop(kwargs.get("run_id"), None)
            if usage_metadata:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)
                if call is not None:
                    details = usage_metadata.get("input_token_details") or {}
                    self.calls.append({
                        **call,
                        "input_tokens": usage_metadata.get("input_tokens", 0),
                        "output_tokens": usage_metadata.get("output_tokens", 0),
                        "cached_input_tokens": details.get("cache_read", 0),
                    })

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }

    def usage_report(self, prices: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Per-call rows and per-model totals, priced from ``prices`` (USD per 1M tokens).

        Cached input tokens are part of ``input_tokens``; they are billed at the
        ``cached_input`` rate and the rest at ``input``. A model missing from the
        table gets a cost of None rather than a guess.
        """
        with self._lock:
            calls = [dict(c) for c in self.calls]
        totals: dict[str, dict[str, Any]] = {}
        for c in calls:
            p = prices.get(c["model"])
            if p:
                cached = c["cached_input_tokens"]
                c["cost_usd"] = round((
                    (c["input_tokens"] - cached) * p["input"]
                    + cached * p.get("cached_input", p["input"])
                    + c["output_tokens"] * p["output"]
                ) / 1e6, 6)
            else:
                c["cost_usd"] = None
            t = totals.setdefault(c["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                               "cached_input_tokens": 0, "cost_usd": 0.0})
            t["calls"] += 1
            for k in ("input_tokens", "output_tokens", "cached_input_tokens"):
                t[k] += c[k]
            t["cost_usd"] = None if t["cost_usd"] is None or c["cost_usd"] is None else round(t["cost_usd"] + c["cost_usd"], 6)
        costs = [t["cost_usd"] for t in totals.values()]
        total_cost = None if None in costs else round(sum(costs), 6)
        return {"calls": calls, "totals_by_model": totals, "total_cost_usd": total_cost}

    def write_usage(self, path: Path, prices: dict[str, dict[str, float]]) -> str:
        """Write usage.json to ``path`` and return a one-line summary."""
        report = self.usage_report(prices)
        Path(path).write_text(json.dumps(report, indent=2))
        tin = sum(t["input_tokens"] for t in report["totals_by_model"].values())
        tout = sum(t["output_tokens"] for t in report["totals_by_model"].values())
        cost = report["total_cost_usd"]
        return (f"LLM usage: {len(report['calls'])} calls, {tin} in / {tout} out tokens, "
                f"cost {'unknown' if cost is None else f'${cost:.4f}'}")
