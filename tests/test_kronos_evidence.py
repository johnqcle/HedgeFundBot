"""Kronos shadow evidence: cutoff, artifacts, and no decision influence.

Network, torch, and the upstream checkout are not required. A real mini-model
smoke lives in test_kronos_inference_smoke.py and is marked integration.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.forecasts import frames, predict
from tradingagents.forecasts.calendar import SessionCalendar
from tradingagents.forecasts.constants import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
)
from tradingagents.forecasts.evaluation import rolling_origin_evaluation
from tradingagents.forecasts.runner import run_kronos_evidence
from tradingagents.reporting import write_report_tree

NY = ZoneInfo("America/New_York")
TRADE_DATE = "2026-08-14"  # Friday
NOW = datetime(2026, 8, 14, 18, 0, tzinfo=NY)


def _sessions(start: str, count: int) -> list:
    return list(pd.bdate_range(start, periods=count).date)


def _bars(start: str = "2026-08-03", count: int = 10, *, extra_future: bool = True) -> pd.DataFrame:
    days = _sessions(start, count)
    close = np.linspace(100.0, 180.0, count)
    # The trade-date close is distinctive so a leaked observation is obvious.
    # The future Monday is a second distinctive value and must not survive the cutoff.
    if days[-1].isoformat() == TRADE_DATE:
        close[-1] = 777.25
    frame = pd.DataFrame({
        "session_date": days,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": np.full(count, 12345.0),
        "adj_close": close - 0.5,
    })
    if extra_future:
        future = frame.iloc[[-1]].copy()
        future["session_date"] = _sessions("2026-08-17", 1)[0]
        future["close"] = 9999.0
        future["open"] = 9998.0
        future["high"] = 10000.0
        future["low"] = 9997.0
        frame = pd.concat([frame, future], ignore_index=True)
    return frame


class _Predictor:
    def __init__(self, value=10.0, fail=False, non_finite=False):
        self.value = value
        self.fail = fail
        self.non_finite = non_finite
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose):
        self.calls.append({
            "pred_len": pred_len, "T": T, "top_k": top_k, "top_p": top_p,
            "sample_count": sample_count, "rows": len(df), "columns": list(df.columns),
        })
        if self.fail:
            raise RuntimeError("boom")
        value = np.inf if self.non_finite else self.value
        data = {name: np.full(pred_len, value) for name in ("open", "high", "low", "close", "volume", "amount")}
        return pd.DataFrame(data, index=pd.RangeIndex(pred_len))


def _config(tmp_path, **overrides) -> dict:
    config = {
        "kronos_enabled": True,
        "kronos_home": str(tmp_path / "KronosOutside"),
        "kronos_influence": "shadow",
        "kronos_attach_to_market_analyst": False,
        "kronos_lookback": 8,
        "kronos_min_bars": 5,
        "kronos_max_context": 2048,
        "kronos_horizons": "2,3",
        "kronos_path_count": 1,
        "kronos_seed": 123,
        "kronos_temperature": 0.7,
        "kronos_top_p": 0.9,
        "kronos_amount_policy": "omit",
        "kronos_calendar": "us_equity_weekday",
        "kronos_timezone": "America/New_York",
        "kronos_device": "cpu",
        "kronos_model_id": MODEL_ID,
        "kronos_model_revision": MODEL_REVISION,
        "kronos_tokenizer_id": TOKENIZER_ID,
        "kronos_tokenizer_revision": TOKENIZER_REVISION,
        "results_dir": str(tmp_path / "results"),
    }
    config.update(overrides)
    return config


def _home(path: Path) -> Path:
    model = path / "model"
    model.mkdir(parents=True)
    (model / "kronos.py").write_text("# upstream stand-in\n", encoding="utf-8")
    return path


def _run(tmp_path, bars=None, predictor=None, **overrides):
    home = _home(tmp_path / "sibling" / "Kronos")
    frame = _bars() if bars is None else bars
    meta = {"adjustment": {"policy": "yahoo_split_adjusted_ohlc_and_volume", "consistent": True,
                           "note": "test fixture"}, "vendor": "fixture"}

    def loader(symbol, as_of_date, now=None):
        return frame.copy(), dict(meta)

    def factory(resolved_home):
        assert resolved_home == home.resolve()
        return predictor or _Predictor()

    evidence = run_kronos_evidence(
        ticker="NVDA", trade_date=TRADE_DATE, asset_type="stock",
        config=_config(tmp_path, kronos_home=str(home), **overrides),
        now=NOW, bars_loader=loader, predictor_factory=factory,
    )
    return evidence, home


@pytest.mark.unit
def test_env_overlay_enables_kronos_without_turning_on_decision_influence(monkeypatch):
    import tradingagents.default_config as default_config

    monkeypatch.setenv("TRADINGAGENTS_KRONOS_ENABLED", "true")
    monkeypatch.setenv("TRADINGAGENTS_KRONOS_HORIZONS", "5,20")
    monkeypatch.setenv("TRADINGAGENTS_KRONOS_LOOKBACK", "128")
    monkeypatch.setenv("TRADINGAGENTS_KRONOS_INFLUENCE", "shadow")
    config = default_config.build_default_config()
    assert config["kronos_enabled"] is True
    assert config["kronos_horizons"] == "5,20"
    assert config["kronos_lookback"] == 128
    assert config["kronos_influence"] == "shadow"
    assert config["kronos_attach_to_market_analyst"] is False


@pytest.mark.unit
def test_disabled_writes_nothing_and_leaves_the_report_alone(tmp_path):
    evidence = run_kronos_evidence(
        ticker="NVDA", trade_date=TRADE_DATE, config={"kronos_enabled": False, "results_dir": str(tmp_path)},
    )
    assert evidence.status == "disabled"
    assert evidence.markdown == ""
    assert evidence.as_state()["kronos_attach_to_market_analyst"] is False
    assert list(tmp_path.iterdir()) == []

    report = write_report_tree({"market_report": "MKT", "kronos_status": "disabled", "kronos_evidence": ""},
                               "NVDA", tmp_path / "report")
    text = report.read_text(encoding="utf-8")
    assert "Kronos" not in text
    assert not (tmp_path / "report" / "6_kronos").exists()


@pytest.mark.unit
def test_forecast_uses_only_completed_bars_and_keeps_observations_separate(tmp_path):
    predictor = _Predictor()
    evidence, _home_path = _run(tmp_path, predictor=predictor)
    assert evidence.status == "ok"
    artifact = Path(evidence.artifact_dir)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    forecast = json.loads((artifact / "forecast.json").read_text(encoding="utf-8"))
    observed = pd.read_csv(artifact / "observed_ohlcv.csv")
    model_input = pd.read_csv(artifact / "model_input.csv")

    assert "2026-08-17" not in set(observed["session_date"])
    assert observed["session_date"].iloc[-1] == TRADE_DATE
    assert "777.25" not in (artifact / "forecast.json").read_text(encoding="utf-8")
    assert "12345" not in (artifact / "forecast.csv").read_text(encoding="utf-8")
    assert "volume" not in model_input.columns
    assert "amount" not in model_input.columns
    assert "adj_close" not in model_input.columns
    assert set(predictor.calls[0]["columns"]) == {"open", "high", "low", "close"}
    assert all(call["sample_count"] == 1 for call in predictor.calls)
    assert [call["pred_len"] for call in predictor.calls] == [2, 3]
    assert forecast["decision_influence"] is False
    assert forecast["rating_changed"] is False
    assert forecast["stops_changed"] is False
    assert forecast["sizing_changed"] is False
    assert forecast["confidence_adjusted"] is False
    assert forecast["pretraining_cutoff"] is None
    assert forecast["pretraining_contamination"] == "unknown"
    assert forecast["model"]["id"] == MODEL_ID
    assert forecast["model"]["revision"] == MODEL_REVISION
    assert forecast["model"]["tokenizer_id"] == TOKENIZER_ID
    assert forecast["model"]["tokenizer_revision"] == TOKENIZER_REVISION
    assert forecast["model"]["weights_bundled"] is False
    assert forecast["seed"] == 123
    assert "Weekdays only" in forecast["calendar_limitation"]
    assert forecast["asof"].startswith("2026-08-14")
    assert forecast["amount"]["volume_source"] == "omitted_to_avoid_synthetic_amount"
    assert forecast["horizons"]["2"]["mean_is_an_uncertainty_interval"] is False
    assert forecast["horizons"]["2"]["dispersion_rows"] is None
    assert forecast["horizons"]["2"]["sampling"]["mode"] == "deterministic_top1"
    assert "probability" in forecast["horizons"]["2"]["dispersion_note"].lower()
    assert manifest["status"] == "ok"
    assert (artifact / "forecast.md").read_text(encoding="utf-8").startswith("# Kronos shadow evidence")
    assert "777.25" in (artifact / "observed_ohlcv.csv").read_text(encoding="utf-8")


@pytest.mark.unit
def test_same_day_bar_is_dropped_until_the_session_closes():
    calendar = SessionCalendar()
    during = datetime(2026, 8, 14, 15, 0, tzinfo=NY)
    bars = _bars(extra_future=False)
    asof = calendar.analysis_asof(TRADE_DATE, during)
    kept, info = frames.completed_history(bars, calendar, asof, lookback=20, min_bars=1)
    assert TRADE_DATE not in set(kept["session_date"].map(lambda day: day.isoformat()))
    assert info["last_completed_session"] == "2026-08-13"
    assert info["dropped"]["future_or_incomplete"] >= 1


@pytest.mark.unit
def test_historical_cutoff_drops_later_sessions_even_if_the_clock_is_later():
    calendar = SessionCalendar()
    later = datetime(2026, 8, 20, 12, 0, tzinfo=NY)
    asof = calendar.analysis_asof(TRADE_DATE, later)
    kept, _info = frames.completed_history(_bars(), calendar, asof, lookback=20, min_bars=1)
    assert kept["session_date"].iloc[-1].isoformat() == TRADE_DATE
    assert asof.date().isoformat() == TRADE_DATE


@pytest.mark.unit
def test_future_sessions_are_weekdays_after_the_last_completed_bar():
    calendar = SessionCalendar()
    days, stamps = frames.future_stamps(datetime(2026, 8, 14).date(), 5, calendar)
    assert [day.isoformat() for day in days] == [
        "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21",
    ]
    assert list(stamps.dt.hour) == [16, 16, 16, 16, 16]
    assert list(stamps.dt.minute) == [0, 0, 0, 0, 0]


@pytest.mark.unit
def test_amount_approximation_is_labeled_and_omit_does_not_invent_it():
    bars = _bars(extra_future=False).iloc[:6].reset_index(drop=True)
    omitted, omitted_meta = frames.model_inputs(bars, "omit")
    assert list(omitted.columns) == ["open", "high", "low", "close"]
    assert omitted_meta["amount_is_approximation"] is False
    assert "synthetic" in omitted_meta["volume_source"]

    approximated, meta = frames.model_inputs(bars, "approximate_volume_times_mean_ohlc")
    expected = bars["volume"] * bars[["open", "high", "low", "close"]].mean(axis=1)
    assert meta["amount_is_approximation"] is True
    assert meta["amount_approximation_formula"] == "volume * mean(open, high, low, close)"
    np.testing.assert_allclose(approximated["amount"], expected)


@pytest.mark.unit
def test_separate_paths_are_not_passed_as_sample_count(tmp_path):
    predictor = _Predictor()
    evidence, _home_path = _run(tmp_path, predictor=predictor, kronos_path_count=3, kronos_horizons="2")
    assert evidence.status == "ok"
    assert len(predictor.calls) == 3
    assert {call["sample_count"] for call in predictor.calls} == {1}
    assert {call["top_k"] for call in predictor.calls} == {0}
    forecast = json.loads((Path(evidence.artifact_dir) / "forecast.json").read_text(encoding="utf-8"))
    horizon = forecast["horizons"]["2"]
    assert horizon["aggregation"] == "arithmetic_mean_of_independent_paths"
    assert horizon["dispersion_label"].startswith("empirical_standard_deviation")
    assert "not a calibrated" in horizon["dispersion_note"].lower()
    assert horizon["same_price_indicator_agreement"]["used_to_adjust_confidence"] is False
    assert (Path(evidence.artifact_dir) / "paths_h2.csv").exists()
    paths = pd.read_csv(Path(evidence.artifact_dir) / "paths_h2.csv")
    assert sorted(paths["path"].unique()) == [0, 1, 2]


@pytest.mark.unit
def test_failure_does_not_write_a_numeric_forecast(tmp_path):
    evidence, _home_path = _run(tmp_path, predictor=_Predictor(fail=True))
    assert evidence.status == "failed"
    artifact = Path(evidence.artifact_dir)
    forecast = json.loads((artifact / "forecast.json").read_text(encoding="utf-8"))
    assert forecast["horizons"]["2"]["rows"] is None
    assert not (artifact / "forecast.csv").exists()
    text = (artifact / "forecast.md").read_text(encoding="utf-8")
    assert "No numeric forecast was produced." in text or "Not produced" in text
    assert "10.0" not in text
    assert "777.25" not in text


@pytest.mark.unit
def test_non_finite_output_is_a_failure(tmp_path):
    evidence, _home_path = _run(tmp_path, predictor=_Predictor(non_finite=True), kronos_horizons="2")
    assert evidence.status == "failed"
    assert "non-finite" in json.loads(
        (Path(evidence.artifact_dir) / "forecast.json").read_text(encoding="utf-8")
    )["horizons"]["2"]["detail"].lower()


@pytest.mark.unit
def test_missing_and_nested_checkouts_are_explicit(tmp_path):
    missing = run_kronos_evidence(
        ticker="NVDA", trade_date=TRADE_DATE, now=NOW,
        config=_config(tmp_path, kronos_home=str(tmp_path / "no-such-kronos"), kronos_min_bars=1),
    )
    assert missing.status == "unavailable"
    assert "kronos_home_missing" in missing.markdown or "No Kronos model package" in missing.markdown
    assert "10.0" not in missing.markdown

    nested = run_kronos_evidence(
        ticker="NVDA", trade_date=TRADE_DATE, now=NOW,
        config=_config(tmp_path, kronos_home=str(Path(__file__).resolve().parents[1] / "tradingagents")),
    )
    assert nested.status == "unavailable"
    assert "nested" in nested.markdown.lower()
    forecast = json.loads((Path(nested.artifact_dir) / "forecast.json").read_text(encoding="utf-8"))
    assert forecast["reason"] == "nested_kronos_checkout"
    assert forecast["horizons"] == {}


@pytest.mark.unit
def test_decision_influence_is_refused(tmp_path):
    evidence = run_kronos_evidence(
        ticker="NVDA", trade_date=TRADE_DATE, now=NOW,
        config=_config(tmp_path, kronos_influence="decision", kronos_attach_to_market_analyst=True),
    )
    assert evidence.status == "unavailable"
    assert evidence.attach is False
    assert "decision" in evidence.markdown.lower()


@pytest.mark.unit
def test_crypto_is_not_run_on_the_equity_calendar(tmp_path):
    evidence = run_kronos_evidence(
        ticker="BTC-USD", trade_date=TRADE_DATE, asset_type="crypto", now=NOW,
        config=_config(tmp_path),
    )
    assert evidence.status == "unavailable"
    body = json.loads((Path(evidence.artifact_dir) / "forecast.json").read_text(encoding="utf-8"))
    assert body["reason"] == "unsupported_asset_type"
    assert body["horizons"] == {}


@pytest.mark.unit
def test_short_history_is_unavailable_rather_than_padded(tmp_path):
    short = _bars(count=3, extra_future=False)
    evidence, _home_path = _run(tmp_path, bars=short, kronos_min_bars=5, kronos_lookback=8)
    assert evidence.status == "unavailable"
    body = json.loads((Path(evidence.artifact_dir) / "forecast.json").read_text(encoding="utf-8"))
    assert body["reason"] == "insufficient_completed_bars"
    assert not (Path(evidence.artifact_dir) / "forecast.csv").exists()


@pytest.mark.unit
def test_market_analyst_prompt_is_unchanged_until_attachment_is_on():
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    from tradingagents.forecasts.analyst import SHADOW_PREAMBLE

    def _invoke(state):
        seen = {}

        def _call(prompt):
            seen["prompt"] = str(prompt)
            return AIMessage(content="market report")

        class _LLM:
            def bind_tools(self, tools):
                return RunnableLambda(_call)

        create_market_analyst(_LLM())(state)
        return seen["prompt"]

    base = {
        "company_of_interest": "NVDA", "trade_date": TRADE_DATE, "asset_type": "stock",
        "instrument_context": "NVDA", "messages": [HumanMessage(content="NVDA")],
        "kronos_status": "ok", "kronos_evidence": "close forecast 10",
        "kronos_attach_to_market_analyst": False,
    }
    hidden = _invoke(base)
    shown = _invoke({**base, "kronos_attach_to_market_analyst": True})
    assert "Kronos" not in hidden
    assert "close forecast 10" not in hidden
    assert SHADOW_PREAMBLE in shown
    assert "close forecast 10" in shown
    assert "Do not change an entry" in shown
    assert "Do not raise confidence" in shown


@pytest.mark.unit
def test_kronos_node_runs_before_the_analysts():
    from langgraph.graph import START

    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.setup import GraphSetup

    class _LLM:
        def with_structured_output(self, *args, **kwargs):
            return self

    workflow = GraphSetup(_LLM(), _LLM(), ConditionalLogic(1, 1), 3).setup_graph(
        ["market"], memory_node=lambda state: {"past_context": "", "memory_note": ""},
    )
    starts = [dst for src, dst in workflow.edges if src == START]
    assert starts == ["Kronos Evidence"]
    assert ("Kronos Evidence", "Market Analyst") in workflow.edges
    assert ("Kronos Evidence", "Memory Log") in workflow.edges
    assert (START, "Market Analyst") not in workflow.edges


@pytest.mark.unit
def test_decision_modules_do_not_read_kronos():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tradingagents/agents/trader/trader.py",
        "tradingagents/agents/managers/portfolio_manager.py",
        "tradingagents/agents/managers/research_manager.py",
        "tradingagents/agents/rating.py",
        "tradingagents/agents/schemas.py",
        "tradingagents/agents/risk_mgmt/aggressive_debator.py",
        "tradingagents/agents/risk_mgmt/conservative_debator.py",
        "tradingagents/agents/risk_mgmt/neutral_debator.py",
    ):
        assert "kronos" not in (root / relative).read_text(encoding="utf-8").lower()


@pytest.mark.unit
def test_report_appendix_carries_shadow_evidence_without_changing_the_market_file(tmp_path):
    out = write_report_tree({
        "market_report": "MKT",
        "trade_date": TRADE_DATE,
        "kronos_status": "ok",
        "kronos_evidence": "# Kronos shadow evidence\n\nStatus: ok\n",
        "final_trade_decision": "**Rating**: Hold",
    }, "NVDA", tmp_path)
    assert (tmp_path / "1_analysts" / "market.md").read_text(encoding="utf-8") == "MKT"
    assert "Kronos shadow evidence" in (tmp_path / "6_kronos" / "evidence.md").read_text(encoding="utf-8")
    complete = out.read_text(encoding="utf-8")
    assert "Kronos shadow evidence: ok" in complete
    assert "not a rating, size, or stop input" in complete


@pytest.mark.unit
def test_rolling_origin_hides_the_future_and_names_the_naive_baseline():
    close = pd.Series([float(i) for i in range(30)], name="close")
    bars = pd.DataFrame({"close": close})
    seen = []

    def predict_close(history, horizon):
        seen.append(history.tolist())
        if history.iloc[-1] != history.tolist()[-1]:
            raise AssertionError("history was reordered")
        return [float(history.iloc[-1])] * horizon

    result = rolling_origin_evaluation(bars, predict_close, [5], min_history=10, origin_step=5)
    assert result["decision_influence"] is False
    assert result["pretraining_contamination"] == "unknown"
    assert any("naive" in caveat.lower() for caveat in result["caveats"])
    assert any("pretraining" in caveat.lower() for caveat in result["caveats"])
    assert result["horizons"]["5"]["n_origins"] > 0
    # A flat forecast matches the naive baseline on a rising line's level error
    # only at the first step; both are scored. The history never includes the last bar.
    assert max(len(history) for history in seen) < len(close)
    assert seen[0][-1] == close.iloc[len(seen[0]) - 1]


@pytest.mark.unit
def test_yahoo_loader_requests_split_consistent_bars_and_drops_the_future(tmp_path, monkeypatch):
    from tradingagents.dataflows.config import run_config
    from tradingagents.dataflows.vendors.yahoo import kronos_bars

    captured = {}
    index = pd.DatetimeIndex(
        ["2026-08-13", "2026-08-14", "2026-08-17"], tz="America/New_York", name="Date",
    )
    downloaded = pd.DataFrame({
        "Open": [1.0, 2.0, 3.0],
        "High": [1.5, 2.5, 3.5],
        "Low": [0.5, 1.5, 2.5],
        "Close": [1.2, 2.2, 9.9],
        "Adj Close": [1.0, 2.0, 9.0],
        "Volume": [10.0, 11.0, 12.0],
    }, index=index)

    class _Ticker:
        def __init__(self, symbol):
            captured["symbol"] = symbol

        def history(self, **kwargs):
            captured["history"] = kwargs
            return downloaded.copy()

    monkeypatch.setattr(kronos_bars.yf, "Ticker", lambda symbol: _Ticker(symbol))
    with run_config({"data_cache_dir": str(tmp_path)}):
        frame, meta = kronos_bars.load_split_consistent_daily_bars("NVDA", "2026-08-14", now=NOW)

    assert captured["history"]["auto_adjust"] is False
    assert captured["history"]["actions"] is False
    assert captured["history"]["end"] == "2026-08-15"
    assert captured["history"]["interval"] == "1d"
    assert list(frame["session_date"].map(lambda day: day.isoformat())) == ["2026-08-13", "2026-08-14"]
    assert frame["close"].iloc[-1] == 2.2
    assert "adj_close" in frame.columns
    assert meta["adjustment"]["auto_adjust"] is False
    assert meta["adjustment"]["consistent"] is True
    assert "9.9" not in frame["close"].astype(str).tolist()


@pytest.mark.unit
def test_seed_formula_is_stable_and_sample_count_stays_one():
    assert predict.seed_for(123, 5, 0) == 5123
    assert predict.seed_for(123, 20, 2) == 20125
    sampling = predict.sampling_parameters(1, temperature=0.2, top_p=0.5)
    assert sampling["sample_count"] == 1
    assert sampling["top_k"] == 1
    assert sampling["mode"] == "deterministic_top1"
    many = predict.sampling_parameters(4, temperature=0.2, top_p=0.5)
    assert many["sample_count"] == 1
    assert many["T"] == 0.2


@pytest.mark.unit
def test_node_fail_open_when_disabled_and_when_enabled_without_a_checkout(tmp_path):
    from tradingagents.dataflows.config import run_config
    from tradingagents.forecasts.node import kronos_evidence_node

    with run_config({"kronos_enabled": False, "results_dir": str(tmp_path)}):
        assert kronos_evidence_node({"company_of_interest": "NVDA", "trade_date": TRADE_DATE})["kronos_status"] == "disabled"
    with run_config({**_config(tmp_path, kronos_home=str(tmp_path / "missing")), "kronos_enabled": True}):
        state = kronos_evidence_node({
            "company_of_interest": "NVDA", "trade_date": TRADE_DATE, "asset_type": "stock",
        })
    assert state["kronos_status"] == "unavailable"
    assert state["kronos_attach_to_market_analyst"] is False
    assert "invent" in state["kronos_evidence"].lower() or "No Kronos" in state["kronos_evidence"]
