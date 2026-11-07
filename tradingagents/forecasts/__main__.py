"""Write a Kronos shadow forecast without starting TradingAgents or calling an LLM.

The AVGO and UI check, after the Windows layout in ``docs/KRONOS.md``::

    python -m tradingagents.forecasts --ticker AVGO --ticker UI --latest-completed --check

That command reads Yahoo daily bars and the local Kronos checkout. It does not
construct the agent graph, call xAI, or send an order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tradingagents.default_config import build_default_config
from tradingagents.forecasts.calendar import CalendarUnavailable, latest_completed_session
from tradingagents.forecasts.runner import run_kronos_evidence
from tradingagents.forecasts.verify import verify_forecast_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write Kronos shadow-forecast artifacts. Does not place a trade or call an LLM."
    )
    parser.add_argument("--ticker", action="append", default=None, help="Repeat for each symbol. Example: --ticker AVGO --ticker UI")
    parser.add_argument("--trade-date", default=None, help="Analysis date YYYY-MM-DD")
    parser.add_argument(
        "--latest-completed",
        action="store_true",
        help="Use the most recent XNYS session whose close is already in the past",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless the artifacts match the XNYS clock and the shadow rules",
    )
    parser.add_argument("--results-dir", default=None, help="Directory for the kronos artifact folders")
    args = parser.parse_args(argv)

    tickers = args.ticker or []
    if not tickers:
        parser.error("pass at least one --ticker")
    if args.latest_completed and args.trade_date:
        parser.error("pass only one of --trade-date and --latest-completed")
    if not args.latest_completed and not args.trade_date:
        parser.error("pass --trade-date or --latest-completed")

    if args.latest_completed:
        try:
            trade_date = latest_completed_session().isoformat()
        except CalendarUnavailable as exc:
            sys.stdout.write(f"{exc.reason}: {exc.detail}\n")
            return 2
        sys.stdout.write(f"Latest completed XNYS session: {trade_date}\n")
    else:
        trade_date = args.trade_date

    config = build_default_config()
    config["kronos_enabled"] = True
    config["kronos_influence"] = "shadow"
    if args.results_dir:
        config["results_dir"] = args.results_dir

    failed = False
    for ticker in tickers:
        evidence = run_kronos_evidence(ticker=ticker, trade_date=trade_date, config=config)
        sys.stdout.write(evidence.markdown)
        if evidence.artifact_dir:
            sys.stdout.write(f"\nArtifacts: {evidence.artifact_dir}\n")
        problems: list[str] = []
        if args.check:
            if not evidence.artifact_dir:
                problems = ["no artifact directory was written"]
            else:
                problems = verify_forecast_dir(Path(evidence.artifact_dir))
            if problems:
                sys.stdout.write(f"Check failed for {ticker}:\n")
                sys.stdout.write("\n".join(f"- {problem}" for problem in problems) + "\n")
            else:
                sys.stdout.write(f"Check passed for {ticker}.\n")
        if evidence.status != "ok" or problems:
            failed = True
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
