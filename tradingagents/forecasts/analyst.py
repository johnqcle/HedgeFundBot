"""The market-analyst gate for Kronos text.

Shadow mode keeps forecast numbers out of the prompt unless attachment is
explicitly enabled. Attachment still forbids using the numbers to change an
entry, a stop, a size, a rating, or a confidence statement.
"""

from __future__ import annotations

SHADOW_PREAMBLE = (
    "KRONOS SHADOW EVIDENCE (non-binding). The block below is quantitative "
    "forecast evidence from Kronos. It does not replace your analysis. "
    "Do not change an entry, a stop, a position size, or a rating because of it. "
    "Do not treat path dispersion as a probability or a calibrated interval. "
    "Do not raise confidence because the forecast agrees with indicators computed "
    "from the same prices; that agreement is not independent evidence. "
    "If the status is not ok, say the forecast is unavailable and do not invent numbers. "
    "If you mention the block, keep it under the heading 'Kronos shadow evidence' "
    "and quote only figures that appear in it."
)


def market_analyst_kronos_context(state: dict) -> str:
    """Prompt text for the market analyst, or empty when Kronos is off or not attached.

    The report and the artifact directory can still carry the evidence when this
    returns empty. That is the shadow default: the decision agents are not given
    the forecast, and the market analyst is not given it either until
    ``kronos_attach_to_market_analyst`` is set.
    """
    if not state.get("kronos_attach_to_market_analyst"):
        return ""
    evidence = (state.get("kronos_evidence") or "").strip()
    if not evidence:
        return ""
    return SHADOW_PREAMBLE + "\n\n" + evidence
