# HedgeFundBot

A multi-agent LLM trading-signal framework powered by xAI's Grok. A team of AI analysts (market, fundamentals, news, sentiment) studies a stock, bull and bear researchers debate it, a trader proposes a plan, and a risk team challenges it before a final rating (Buy / Overweight / Hold / Underweight / Sell) with entry, trim, and stop levels.

> **Not financial advice.** This is a research tool. Its output has not been backtested to show an edge, can be wrong, and should never be traded on blindly.

## Built on TradingAgents

HedgeFundBot is a modified version of **[TradingAgents](https://github.com/TauricResearch/TradingAgents)** by [Tauric Research](https://github.com/TauricResearch), released under the Apache 2.0 License (see `LICENSE`). All credit for the core multi-agent design (analyst team, researcher debate, trader, risk management and portfolio manager) goes to the TradingAgents authors. Their original README is kept as [`TRADINGAGENTS_README.md`](TRADINGAGENTS_README.md). If you use this work in research, please cite their paper as described there.

## What's different from TradingAgents

| Area | TradingAgents | HedgeFundBot |
| --- | --- | --- |
| Default models | OpenAI | Set up for xAI Grok (`grok-4.6` deep thinking, `grok-4.3` quick thinking) via `.env` |
| Social sentiment | Generic news/social sources | Adds **X (Twitter) posts** as a sentiment source (`tradingagents/dataflows/vendors/x_posts.py`), read from a local cache file. Posts are cut off at the market close of the trade date, so no future data leaks in |
| Cost tracking | Call and token counts | Writes a **`usage.json`** per run with tokens and estimated USD cost by model and by agent step, including cached-input pricing |
| Runaway protection | none | **LLM call cap** per run (`TRADINGAGENTS_MAX_LLM_CALLS`, default 150) that stops a looping run |
| Data caching | — | **Tool cache** reuses market-data pulls for past dates (`TRADINGAGENTS_TOOL_CACHE`) |
| Token savings | — | Price data drops empty dividend/split columns, skips non-trading days, and rounds indicators to 6 significant digits, cutting prompt size and cost |
| Tests | Upstream suite | Adds tests for X posts, usage logging and token savings |

## Install

Requires Python 3.11+ and an [xAI API key](https://console.x.ai/).

```bash
git clone https://github.com/johnqcle/HedgeFundBot.git
cd HedgeFundBot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env             # Windows: copy .env.example .env
```

## Add your API key

Your key is **never** stored in this repo. Put it in `.env` (`XAI_API_KEY=...`) or set it as an environment variable:

```bash
export XAI_API_KEY="your-key-here"         # Windows PowerShell: $env:XAI_API_KEY="your-key-here"
```

Then set the model options in `.env`:

```
TRADINGAGENTS_LLM_PROVIDER=xai
TRADINGAGENTS_DEEP_THINK_LLM=grok-4.6
TRADINGAGENTS_QUICK_THINK_LLM=grok-4.3
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
TRADINGAGENTS_MAX_RISK_ROUNDS=1
```

`.env` is in `.gitignore`, so it won't be committed if you put a key there.

## Run

```bash
tradingagents               # interactive: pick ticker, date, analysts
```

Reports are saved under `~/.tradingagents/logs/reports/<TICKER>_<date>/`, including `complete_report.md` and `usage.json`.

### Optional: X sentiment data
Save posts for a ticker and trade date to `x_cache/<TICKER>_<YYYY-MM-DD>.json` (format documented at the top of `x_posts.py`), or point `X_CACHE_DIR` elsewhere. If no file exists, the sentiment analyst is told the X data is missing and carries on.

## License

Apache 2.0, same as TradingAgents. See `LICENSE`.
