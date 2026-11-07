# Kronos shadow forecasts

Kronos ([shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), MIT, Copyright (c) 2025 ShiYu) can add a quantitative price-path forecast to a HedgeFundBot run. The forecast is evidence. It does not replace the market, sentiment, news, or fundamentals analysts, the bull/bear debate, the trader, or the risk team. It does not change a rating, a position size, or a stop.

This file is the setup for the intended machine `DESKTOP-14USNKJ`, with the bot at `C:\Users\johnq\Projects\HedgeFundBot`. The integration was prepared on a branch of [johnqcle/HedgeFundBot](https://github.com/johnqcle/HedgeFundBot). It does not copy files onto that machine and it does not install Kronos there.

## Where the upstream code belongs

Clone Kronos **beside** HedgeFundBot:

```text
C:\Users\johnq\Projects\HedgeFundBot
C:\Users\johnq\Projects\Kronos
```

Do not clone it inside the bot tree. These paths are refused, and the root ones are gitignored so a nested checkout is not committed:

- `C:\Users\johnq\Projects\HedgeFundBot\Kronos`
- `C:\Users\johnq\Projects\HedgeFundBot\vendor\Kronos`
- `C:\Users\johnq\Projects\HedgeFundBot\third_party\Kronos`
- `C:\Users\johnq\Projects\HedgeFundBot\tradingagents\...` (anything under the package)

A nested checkout is a second git repository inside this one, and a copy under `tradingagents/` would be installed with the package. Weights are not in this repository. The first forecast downloads the pinned Hugging Face revisions into the local Hugging Face cache.

Reviewed interface commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`. At that commit, `KronosPredictor.predict(..., sample_count=N)` draws N paths and returns their average. It does not return an uncertainty interval. This integration always calls `predict` with `sample_count=1`. Dispersion, if you set a path count above 1, is the sample standard deviation across separately seeded calls and is labeled as empirical model dispersion, not a calibrated probability.

## Windows setup

PowerShell, from a new terminal:

```powershell
cd C:\Users\johnq\Projects
git clone https://github.com/shiyu-coder/Kronos.git

cd C:\Users\johnq\Projects\HedgeFundBot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[kronos]"
```

Do **not** run `pip install -r C:\Users\johnq\Projects\Kronos\requirements.txt` inside this virtual environment. That file pins `pandas==2.2.2`. HedgeFundBot requires pandas 3. The optional extra installs the libraries Kronos imports (torch, einops, huggingface_hub, safetensors, tqdm) without downgrading pandas.

Kronos itself is MIT. HedgeFundBot stays Apache-2.0. The MIT notice is recorded on every manifest and is not replaced by this repository's license.

Add the following to `C:\Users\johnq\Projects\HedgeFundBot\.env` (the file is gitignored):

```text
TRADINGAGENTS_KRONOS_ENABLED=true
TRADINGAGENTS_KRONOS_HOME=C:\Users\johnq\Projects\Kronos
TRADINGAGENTS_KRONOS_INFLUENCE=shadow
TRADINGAGENTS_KRONOS_ATTACH=false
TRADINGAGENTS_KRONOS_DEVICE=cpu
TRADINGAGENTS_KRONOS_HORIZONS=5,20
TRADINGAGENTS_KRONOS_LOOKBACK=256
TRADINGAGENTS_KRONOS_MIN_BARS=64
TRADINGAGENTS_KRONOS_SEED=123
TRADINGAGENTS_KRONOS_PATH_COUNT=1
TRADINGAGENTS_KRONOS_AMOUNT_POLICY=omit
TRADINGAGENTS_KRONOS_CALENDAR=us_equity_weekday
TRADINGAGENTS_KRONOS_TIMEZONE=America/New_York
```

`TRADINGAGENTS_KRONOS_ATTACH=false` keeps the forecast out of the market-analyst prompt. The report still gets an appendix, and the artifacts below are still written. Set it to `true` only when you want the analyst to see the block. The block tells the analyst not to change an entry, a stop, a size, a rating, or a confidence statement. Later agents still read the market report, so pasted numbers can travel. Leave attachment off until the evaluation below has been run.

`kronos_influence` accepts only `shadow`. Any other value is an unavailable status and produces no forecast.

Pinned model (override only if you know the pair):

| Role | Hugging Face id | Revision |
| --- | --- | --- |
| Model | `NeoQuasar/Kronos-mini` | `f4e68697d9d5aed55cef5c96aabc3376bcad9f81` |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-2k` | `26966d0035065a0cae0ebad7af8ece35bc1fb51c` |

Mini is the 4.1M model. Its tokenizer is Tokenizer-2k, context 2048. Tokenizer-base belongs to Kronos-small and Kronos-base and is the wrong tokenizer for mini.

## Run a forecast without the agents

This command does not call an LLM, a broker, or the xAI API. It does download Yahoo daily bars and, the first time, the pinned weights.

```powershell
cd C:\Users\johnq\Projects\HedgeFundBot
.\.venv\Scripts\Activate.ps1
python -m tradingagents.forecasts --ticker NVDA --trade-date 2026-09-01
```

Artifacts land under the results directory (default `%USERPROFILE%\.tradingagents\logs`):

```text
%USERPROFILE%\.tradingagents\logs\kronos\NVDA_2026-09-01\
    manifest.json
    observed_ohlcv.csv
    model_input.csv
    forecast.json
    forecast.csv
    forecast.md
    paths_h5.csv          # only when TRADINGAGENTS_KRONOS_PATH_COUNT is greater than 1
    paths_h20.csv
```

`observed_ohlcv.csv` is the history that survived the cutoff. `model_input.csv` is the frame passed to `predict`, including the session-close timestamps. `forecast.json` and `forecast.csv` contain forecast rows only. They are not appended to the observed file.

A normal `tradingagents` run, with Kronos enabled, writes the same folder and adds section "VI. Kronos shadow evidence" to the saved report. The graph step runs before the analysts. If Kronos cannot run, the status is `unavailable` or `failed`, the other agents still run, and no numeric forecast is written.

Exit code 0 means a forecast was produced. Exit code 2 means it was not. Read `forecast.md` for the reason. Do not fill in a price from a failed run.

## What the data is

- Timezone `America/New_York`. A daily bar is complete at 16:00. The analysis as-of is the end of the trade date, and never later than the clock. Same-day bars are dropped until that close. Bars after the trade date are dropped.
- Calendar `us_equity_weekday`: weekdays only. Holidays and early closes are not modeled. The limitation is copied into every manifest. Forecast timestamps can fall on a closed session.
- Yahoo is requested with `auto_adjust=False` and an exclusive end the day after the trade date. Open/High/Low/Close stay on Yahoo's split-adjusted close. Volume is the share volume returned with that close. `Adj Close` is stored on the observed file and is not a model input. This is deliberate: the indicator feed uses `auto_adjust=True`, which dividend-adjusts prices and leaves volume unchanged, so that series is not corporate-action-consistent with volume.
- Amount is not on Yahoo daily bars. The default policy `omit` does not pass volume either, because upstream `predict` would silently set `amount = volume * mean(open, high, low, close)`. Observed volume remains in `observed_ohlcv.csv`. Set `TRADINGAGENTS_KRONOS_AMOUNT_POLICY=approximate_volume_times_mean_ohlc` only when you want that formula, and the manifest labels it as an approximation. An observed `amount` column, if a future loader supplies one, is passed through and labeled `observed`.
- One path (`KRONOS_PATH_COUNT=1`) uses top-1 sampling and the configured seed. It is one draw, not a confidence band. A path count from 2 to 8 stores each path and reports the arithmetic mean plus the sample standard deviation. That spread is not a probability.
- The pretraining sample end date is **unknown**. The paper (arXiv:2508.02739), the upstream README, and the Kronos-mini model card do not publish one. Every manifest sets `pretraining_contamination` to `unknown`. A backtest on history the model may have trained on is not out-of-sample.

Crypto and other asset types are refused. The equity calendar is not applied to them.

## Rolling-origin evaluation, before any decision influence

Do not wire Kronos into ratings, size, or stops until this has been run and read. The scorer is `tradingagents.forecasts.evaluation.rolling_origin_evaluation`. It does not download data and it does not load the model. You pass bars that were already cut at each origin, and a function that sees only the close history through that origin.

For each origin `t` and horizon `h` in {5, 20}:

1. History is the completed closes through `t` only.
2. The model forecast is whatever that function returns for `h` steps.
3. The naive baseline repeats the last completed close for `h` steps.
4. Record MAE and RMSE of the close path, and whether the sign of the move from the last close to step `h` matches the realized sign.
5. Compare the model to the naive baseline on the same origins. A win on one window is not a change to a rating.

The result always includes these caveats:

- The history has to contain only bars that had closed at that origin. The scorer will not see later rows, but it cannot repair a frame that was built with them.
- Beating the flat close is not permission to change a rating, a size, or a stop.
- Pretraining contamination is unknown, so scores on older history are not out-of-sample skill.
- Re-downloading split-adjusted Yahoo history can revise prices after the origin date.
- The weekday calendar misses holidays and early closes.
- Sample dispersion is not a calibrated probability and is not part of the score.
- Agreement with RSI, MACD, or any other indicator computed from the same prices is not independent confirmation and must not raise confidence.

Decision influence stays unimplemented on purpose. `kronos_influence=decision` is rejected.

## Smoke test

The default `pytest` run does not download weights. To exercise the pinned mini model and Tokenizer-2k on a synthetic window (no API key, no broker):

```powershell
$env:KRONOS_SMOKE_HOME = "C:\Users\johnq\Projects\Kronos"
pytest -m integration tests/test_kronos_inference_smoke.py
```

`KRONOS_SMOKE_HOME` is not a `TRADINGAGENTS_*` variable. The test suite clears those so a local `.env` cannot change defaults.
