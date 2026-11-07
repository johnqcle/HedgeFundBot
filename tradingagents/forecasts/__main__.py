"""Write a Kronos shadow forecast without starting TradingAgents or calling an LLM.

Example (after the Windows layout in docs/KRONOS.md)::

    python -m tradingagents.forecasts --ticker NVDA --trade-date 2026-09-01
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.default_config import build_default_config
from tradingagents.forecasts.runner import run_kronos_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Kronos shadow-forecast artifacts. Does not place a trade.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--trade-date", required=True, help="Analysis date YYYY-MM-DD")
    parser.add_argument("--results-dir", default=None, help="Directory for the kronos artifact folder")
    args = parser.parse_args(argv)

    config = build_default_config()
    config["kronos_enabled"] = True
    config["kronos_influence"] = "shadow"
    if args.results_dir:
        config["results_dir"] = args.results_dir

    evidence = run_kronos_evidence(ticker=args.ticker, trade_date=args.trade_date, config=config)
    sys.stdout.write(evidence.markdown)
    if evidence.artifact_dir:
        sys.stdout.write(f"\nArtifacts: {evidence.artifact_dir}\n")
    return 0 if evidence.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
