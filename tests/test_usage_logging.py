"""Per-call token and cost rows from the stats callback, read from a mocked
response's usage_metadata (never estimated)."""

from __future__ import annotations

import json
import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cli.stats_handler import StatsCallbackHandler

PRICES = {"grok-4.3": {"input": 1.25, "cached_input": 0.20, "output": 2.50}}


def _call(handler, node, model, usage):
    run_id = uuid.uuid4()
    handler.on_chat_model_start({}, [[]], run_id=run_id,
                                metadata={"langgraph_node": node},
                                invocation_params={"model": model})
    msg = AIMessage(content="ok", usage_metadata=usage)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run_id)


@pytest.mark.unit
def test_usage_rows_totals_and_cost(tmp_path):
    h = StatsCallbackHandler()
    _call(h, "Sentiment Analyst", "grok-4.3",
          {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000,
           "input_token_details": {"cache_read": 400_000}})
    _call(h, "Trader", "mystery-model",
          {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    summary = h.write_usage(tmp_path / "usage.json", PRICES)
    report = json.loads((tmp_path / "usage.json").read_text())

    first = report["calls"][0]
    assert first["node"] == "Sentiment Analyst" and first["cached_input_tokens"] == 400_000
    # 600k uncached * 1.25 + 400k cached * 0.20 + 1M out * 2.50
    assert first["cost_usd"] == pytest.approx(0.75 + 0.08 + 2.50)
    assert report["calls"][1]["cost_usd"] is None
    assert report["totals_by_model"]["grok-4.3"]["calls"] == 1
    assert report["total_cost_usd"] is None
    assert "2 calls" in summary and "cost unknown" in summary
