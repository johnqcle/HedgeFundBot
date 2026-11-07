"""Call Kronos one path at a time.

``KronosPredictor.predict(..., sample_count=N)`` draws N paths and returns
their mean. The mean is computed in normalized space and then denormalized;
that is an average path, not a quantile and not an interval. This wrapper
always passes ``sample_count=1``. Dispersion, when requested, is the sample
standard deviation across separately seeded calls, and it is labeled as
empirical model dispersion rather than a calibrated probability.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

PRICE_AND_SIZE = ["open", "high", "low", "close", "volume", "amount"]

DISPERSION_LABEL = "empirical_standard_deviation_across_independently_seeded_paths"
DISPERSION_NOTE = (
    "Sample standard deviation across separately seeded Kronos paths. "
    "Not a calibrated prediction interval and not a probability. "
    "Upstream sample_count averages paths and would have hidden them."
)
SAME_PRICE_INDICATORS = {
    "used_to_adjust_confidence": False,
    "confidence_adjusted": False,
    "reason": (
        "Indicators computed from the same OHLCV are not independent of the "
        "forecast. Agreement is not measured and does not raise confidence."
    ),
}


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch when torch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sampling_parameters(path_count: int, temperature: float, top_p: float) -> dict:
    """Deterministic top-1 for a single path. Independent samples only when paths are kept."""
    if path_count == 1:
        return {
            "T": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_count": 1,
            "mode": "deterministic_top1",
            "temperature_ignored": temperature,
        }
    return {
        "T": float(temperature),
        "top_k": 0,
        "top_p": float(top_p),
        "sample_count": 1,
        "mode": "independent_paths",
        "temperature_ignored": None,
    }


def seed_for(base_seed: int, horizon: int, path_index: int) -> int:
    """A recorded seed per horizon and path. Horizons do not share a running RNG."""
    return int(base_seed) + int(horizon) * 1000 + int(path_index)


def forecast_horizon(
    predictor,
    model_frame: pd.DataFrame,
    history_stamps: pd.Series,
    future_stamps: pd.Series,
    *,
    horizon: int,
    base_seed: int,
    path_count: int,
    temperature: float,
    top_p: float,
) -> dict:
    """One horizon: ``path_count`` calls with ``sample_count=1``, then an optional mean."""
    if len(future_stamps) != horizon:
        raise ValueError(f"future timestamps ({len(future_stamps)}) must equal horizon {horizon}")
    if len(history_stamps) != len(model_frame):
        raise ValueError("history timestamps must match the model frame row for row")

    sampling = sampling_parameters(path_count, temperature, top_p)
    paths: list[pd.DataFrame] = []
    seeds: list[int] = []
    for index in range(path_count):
        seed = seed_for(base_seed, horizon, index)
        seeds.append(seed)
        set_deterministic_seed(seed)
        predicted = predictor.predict(
            df=model_frame,
            x_timestamp=history_stamps,
            y_timestamp=future_stamps,
            pred_len=horizon,
            T=sampling["T"],
            top_k=sampling["top_k"],
            top_p=sampling["top_p"],
            sample_count=1,
            verbose=False,
        )
        frame = _align_prediction(predicted, future_stamps)
        paths.append(frame)

    mean = _average_paths(paths)
    dispersion = _dispersion(paths) if path_count > 1 else None
    return {
        "status": "ok",
        "horizon": horizon,
        "pred_len": horizon,
        "seed_base": base_seed,
        "seeds": seeds,
        "path_count": path_count,
        "sample_count_passed_to_predict": 1,
        "sampling": sampling,
        "aggregation": "single_path" if path_count == 1 else "arithmetic_mean_of_independent_paths",
        "mean_is_an_uncertainty_interval": False,
        "dispersion_label": DISPERSION_LABEL if dispersion is not None else None,
        "dispersion_note": DISPERSION_NOTE if dispersion is not None else (
            "One path was drawn. No dispersion is reported, and this is not a "
            "probability or a calibrated interval. sample_count was 1, so predict() "
            "did not average paths."
        ),
        "same_price_indicator_agreement": dict(SAME_PRICE_INDICATORS),
        "mean": mean,
        "dispersion": dispersion,
        "paths": paths,
    }


def _align_prediction(predicted: pd.DataFrame, future_stamps: pd.Series) -> pd.DataFrame:
    frame = predicted.copy()
    missing = [name for name in PRICE_AND_SIZE if name not in frame.columns]
    if missing:
        raise ValueError(f"Kronos prediction missing columns {missing}")
    if len(frame) != len(future_stamps):
        raise ValueError(
            f"Kronos returned {len(frame)} rows for {len(future_stamps)} future sessions"
        )
    values = frame.loc[:, PRICE_AND_SIZE].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Kronos returned a non-finite value")
    # The upstream index follows whatever object was passed as y_timestamp.
    # Rebind it to the session closes we asked for so a Series index cannot
    # replace those timestamps.
    out = pd.DataFrame(values, columns=PRICE_AND_SIZE)
    out.index = pd.DatetimeIndex(pd.to_datetime(future_stamps.to_numpy()))
    return out


def _average_paths(paths: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = np.stack([path.to_numpy(dtype=float) for path in paths], axis=0)
    mean = stacked.mean(axis=0)
    out = pd.DataFrame(mean, columns=PRICE_AND_SIZE, index=paths[0].index)
    return out


def _dispersion(paths: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = np.stack([path.to_numpy(dtype=float) for path in paths], axis=0)
    # ddof=1: a sample standard deviation. Still not a calibrated interval.
    spread = stacked.std(axis=0, ddof=1)
    return pd.DataFrame(spread, columns=PRICE_AND_SIZE, index=paths[0].index)
