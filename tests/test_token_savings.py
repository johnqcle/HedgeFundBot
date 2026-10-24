"""LLM-call cap and point-in-time tool cache."""
import pytest

from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.graph.trading_graph import LLMCallLimiter, LLMCallLimitExceeded


def test_llm_call_limiter_stops_past_limit():
    limiter = LLMCallLimiter(2)
    limiter.on_chat_model_start({}, [])
    limiter.on_chat_model_start({}, [])
    with pytest.raises(LLMCallLimitExceeded):
        limiter.on_chat_model_start({}, [])


def test_llm_call_limiter_zero_means_no_cap():
    limiter = LLMCallLimiter(0)
    for _ in range(500):
        limiter.on_chat_model_start({}, [])


def _stub(monkeypatch, tmp_path):
    set_config({"tool_cache_enabled": True, "data_cache_dir": str(tmp_path),
                "tool_vendors": {"get_stock_data": "yfinance"}})
    calls = []

    def fetch(symbol, start, end):
        calls.append((symbol, start, end))
        return f"data {symbol} {start} {end}"

    monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "yfinance", fetch)
    return calls


def test_past_dated_pull_is_cached_per_exact_args(monkeypatch, tmp_path):
    calls = _stub(monkeypatch, tmp_path)
    assert router.route_to_vendor("get_stock_data", "AVGO", "2024-01-01", "2024-02-01") == "data AVGO 2024-01-01 2024-02-01"
    router.route_to_vendor("get_stock_data", "AVGO", "2024-01-01", "2024-02-01")
    assert len(calls) == 1
    # A different as-of date (earlier or later) never reuses that pull.
    assert router.route_to_vendor("get_stock_data", "AVGO", "2024-01-01", "2024-01-15") == "data AVGO 2024-01-01 2024-01-15"
    router.route_to_vendor("get_stock_data", "NVDA", "2024-01-01", "2024-02-01")
    assert len(calls) == 3


def test_today_or_undated_pull_is_not_cached(monkeypatch, tmp_path):
    from datetime import date
    calls = _stub(monkeypatch, tmp_path)
    today = date.today().isoformat()
    router.route_to_vendor("get_stock_data", "AVGO", "2024-01-01", today)
    router.route_to_vendor("get_stock_data", "AVGO", "2024-01-01", today)
    assert len(calls) == 2
