"""Rolling-origin scores against a naive baseline.

This module does not download prices, load Kronos, or change a rating. It
scores a close path the caller already produced from history that was complete
at each origin. The caveats are part of the result so a score cannot be
separated from the leakage limits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from tradingagents.forecasts.constants import PRETRAINING_NOTE

CAVEATS: tuple[str, ...] = (
    "Each origin may use only bars whose session had closed at that origin. "
    "The scorer passes the close history through that row and no later row; "
    "the caller still has to have built ``bars`` that way.",
    "The naive baseline repeats the last observed close. Beating it on one "
    "window is not a reason to change a rating, a size, or a stop.",
    PRETRAINING_NOTE,
    "Split-adjusted history can be revised after the origin date. A backtest "
    "that re-downloads it is not the series that existed at the cutoff.",
    "Origins and forecast stamps have to follow the exchange schedule (XNYS holidays and early closes). A weekday grid puts a timestamp on a closed session.",
    "Dispersion across Kronos samples is not a calibrated probability and is not an input to these scores.",
    "Agreement between a Kronos path and an indicator computed from the same prices is not independent evidence and does not raise confidence.",
    "Decision influence stays off until a rolling-origin result is produced under these rules on data after a known pretraining cutoff.",
)


def naive_flat_close(last_close: float, horizon: int) -> list[float]:
    """The random-walk baseline: the last completed close, repeated."""
    return [float(last_close)] * horizon


def score_close_path(
    predicted: Sequence[float],
    realized: Sequence[float],
    *,
    last_close: float,
) -> dict:
    """Level errors and end-point direction versus the last observed close."""
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realized, dtype=float)
    if pred.shape != real.shape or pred.ndim != 1 or pred.size == 0:
        raise ValueError("predicted and realized closes must be the same non-empty length")
    if not np.isfinite(pred).all() or not np.isfinite(real).all():
        raise ValueError("closes must be finite")
    error = pred - real
    pred_move = float(pred[-1] - last_close)
    real_move = float(real[-1] - last_close)
    if pred_move == 0 or real_move == 0:
        direction = "flat"
    else:
        direction = "match" if (pred_move > 0) == (real_move > 0) else "miss"
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "end_direction": direction,
        "end_predicted_minus_last": pred_move,
        "end_realized_minus_last": real_move,
    }


def rolling_origin_evaluation(
    bars: pd.DataFrame,
    predict_close: Callable[[pd.Series, int], Sequence[float]],
    horizons: Sequence[int],
    *,
    min_history: int,
    origin_step: int = 1,
) -> dict:
    """Score ``predict_close`` at each origin against the flat-close baseline.

    ``bars`` is oldest-first and needs a ``close`` column. At an origin whose
    last history row is ``t``, the callable receives ``close.iloc[: t + 1]``
    only. Realized closes are the next ``horizon`` rows. Origins that do not
    leave a full realized path are not scored.
    """
    if "close" not in bars.columns:
        raise ValueError("bars need a close column")
    if min_history < 1 or origin_step < 1:
        raise ValueError("min_history and origin_step must be positive")
    horizons = [int(h) for h in horizons]
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must be positive")

    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    if close.isna().any():
        raise ValueError("close has gaps; the scorer does not fill them")

    per_horizon: dict[str, dict] = {}
    n = int(len(close))
    for horizon in horizons:
        model_scores = []
        naive_scores = []
        origins = []
        last_start = n - horizon
        for t in range(min_history - 1, last_start, origin_step):
            history = close.iloc[: t + 1]
            realized = close.iloc[t + 1 : t + 1 + horizon]
            predicted = list(predict_close(history.copy(), horizon))
            if len(predicted) != horizon:
                raise ValueError(
                    f"predict_close returned {len(predicted)} steps for horizon {horizon}"
                )
            last = float(history.iloc[-1])
            model_scores.append(score_close_path(predicted, realized, last_close=last))
            naive_scores.append(
                score_close_path(naive_flat_close(last, horizon), realized, last_close=last)
            )
            origins.append(int(t))
        per_horizon[str(horizon)] = _summarize(horizon, origins, model_scores, naive_scores)

    return {
        "method": "rolling_origin_close_vs_naive_flat",
        "decision_influence": False,
        "pretraining_contamination": "unknown",
        "caveats": list(CAVEATS),
        "origins_rule": (
            "history is close.iloc[:origin + 1]; realized is the next horizon closes; "
            "the naive baseline repeats the last history close"
        ),
        "horizons": per_horizon,
    }


def _summarize(horizon: int, origins: list[int], model_scores: list[dict], naive_scores: list[dict]) -> dict:
    if not model_scores:
        return {
            "horizon": horizon,
            "n_origins": 0,
            "model": None,
            "naive_flat_close": None,
            "note": "no origin had enough history and a full realized path",
        }

    def _mean(rows: list[dict], key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    def _direction_rate(rows: list[dict]) -> dict:
        labels = [row["end_direction"] for row in rows]
        counted = [label for label in labels if label in ("match", "miss")]
        matches = sum(label == "match" for label in counted)
        return {
            "match": matches,
            "miss": sum(label == "miss" for label in counted),
            "flat": sum(label == "flat" for label in labels),
            "match_rate_excluding_flats": (matches / len(counted)) if counted else None,
        }

    return {
        "horizon": horizon,
        "n_origins": len(origins),
        "origin_last_index": origins,
        "model": {
            "mae": _mean(model_scores, "mae"),
            "rmse": _mean(model_scores, "rmse"),
            "end_direction": _direction_rate(model_scores),
        },
        "naive_flat_close": {
            "mae": _mean(naive_scores, "mae"),
            "rmse": _mean(naive_scores, "rmse"),
            "end_direction": _direction_rate(naive_scores),
        },
    }
