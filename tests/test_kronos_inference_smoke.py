"""Real Kronos-mini inference, only when a checkout and torch are already present.

Not part of the default suite (`pytest` excludes the integration marker). It
does not use an API key. Weights download from Hugging Face the first time.

    set KRONOS_SMOKE_HOME=C:\\Users\\johnq\\Projects\\Kronos
    pytest -m integration tests/test_kronos_inference_smoke.py

The variable is not a ``TRADINGAGENTS_*`` setting: the test suite blanks those
so a local ``.env`` cannot change defaults.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.smoke
def test_kronos_mini_smoke_on_a_synthetic_window(tmp_path):
    home = os.environ.get("KRONOS_SMOKE_HOME", "")
    if not home or not (Path(home) / "model" / "kronos.py").is_file():
        pytest.skip("Set KRONOS_SMOKE_HOME to a Kronos checkout outside this repo")
    pytest.importorskip("torch")

    from tradingagents.forecasts.runner import run_kronos_evidence

    # Ten completed sessions is below the production minimum on purpose: the
    # smoke checks that the pinned mini model and Tokenizer-2k return a finite
    # path, not that the path is a useful forecast.
    days = list(pd.bdate_range("2024-01-02", periods=12).date)
    close = 100 + np.linspace(0, 3, len(days))
    bars = pd.DataFrame({
        "session_date": days,
        "open": close - 0.4,
        "high": close + 0.6,
        "low": close - 0.8,
        "close": close,
        "volume": np.full(len(days), 1_000.0),
    })

    def loader(symbol, as_of_date, now=None):
        return bars.copy(), {
            "vendor": "synthetic-smoke",
            "adjustment": {
                "policy": "synthetic_unadjusted",
                "consistent": True,
                "note": "Synthetic smoke bars, not a market download.",
            },
        }

    config = {
        "kronos_enabled": True,
        "kronos_home": home,
        "kronos_influence": "shadow",
        "kronos_attach_to_market_analyst": False,
        "kronos_lookback": 12,
        "kronos_min_bars": 8,
        "kronos_max_context": 2048,
        "kronos_horizons": "2",
        "kronos_path_count": 1,
        "kronos_seed": 123,
        "kronos_device": "cpu",
        "kronos_amount_policy": "omit",
        "kronos_calendar": "us_equity_weekday",
        "kronos_timezone": "America/New_York",
        "results_dir": str(tmp_path),
    }
    evidence = run_kronos_evidence(
        ticker="SMOKE",
        trade_date="2024-01-17",
        config=config,
        now=datetime(2024, 1, 17, 18, 0, tzinfo=ZoneInfo("America/New_York")),
        bars_loader=loader,
    )
    assert evidence.status == "ok", evidence.markdown
    forecast = (Path(evidence.artifact_dir) / "forecast.json").read_text(encoding="utf-8")
    assert "NeoQuasar/Kronos-mini" in forecast
    assert "Kronos-Tokenizer-2k" in forecast
    assert evidence.attach is False
