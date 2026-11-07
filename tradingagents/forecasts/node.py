"""Graph step that files Kronos shadow evidence before the analysts run.

Disabled runs return immediately. A failure is a status on the state, not an
exception, so the analyst, debate, and risk steps still run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def kronos_evidence_node(state: dict) -> dict:
    from tradingagents.dataflows.config import get_config

    config = get_config()
    if not config.get("kronos_enabled"):
        return _disabled()
    try:
        from tradingagents.forecasts.runner import run_kronos_evidence

        evidence = run_kronos_evidence(
            ticker=state.get("company_of_interest") or "",
            trade_date=str(state.get("trade_date") or ""),
            asset_type=state.get("asset_type") or "stock",
            config=config,
        )
        return evidence.as_state()
    except Exception as exc:
        logger.warning("Kronos evidence node failed open: %s", exc)
        return {
            "kronos_status": "failed",
            "kronos_evidence": (
                "# Kronos shadow evidence\n\n"
                f"Status: failed\n\nNo forecast was produced ({type(exc).__name__}). "
                "Do not invent prices, a rating, a size, or a stop.\n"
            ),
            "kronos_artifact_dir": "",
            "kronos_attach_to_market_analyst": False,
        }


def _disabled() -> dict:
    return {
        "kronos_status": "disabled",
        "kronos_evidence": "",
        "kronos_artifact_dir": "",
        "kronos_attach_to_market_analyst": False,
    }
