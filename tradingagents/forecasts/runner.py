"""Build one shadow Kronos run and write its artifacts.

Any failure becomes an explicit status and an empty forecast. Numbers are
written only for horizons that actually returned a finite path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.forecasts import artifacts, frames, predict
from tradingagents.forecasts.calendar import CALENDAR_LIMITATION, require_calendar
from tradingagents.forecasts.constants import (
    AMOUNT_POLICY_OMIT,
    CALENDAR_ID,
    DEFAULT_DEVICE,
    DEFAULT_HORIZONS,
    DEFAULT_LOOKBACK,
    DEFAULT_MIN_BARS,
    DEFAULT_PATH_COUNT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEZONE,
    DEFAULT_TOP_P,
    INFLUENCE_SHADOW,
    KRONOS_REPO,
    LICENSE_NAME,
    LICENSE_NOTICE,
    MAX_CONTEXT,
    MAX_PATH_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    PAPER_URL,
    PRETRAINING_CONTAMINATION,
    PRETRAINING_CUTOFF,
    PRETRAINING_NOTE,
    REVIEWED_UPSTREAM_COMMIT,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
)
from tradingagents.forecasts.upstream import (
    KronosUpstreamError,
    commit_warning,
    load_predictor,
    resolve_kronos_home,
    upstream_commit,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KronosEvidence:
    status: str
    markdown: str
    artifact_dir: str
    attach: bool

    def as_state(self) -> dict:
        return {
            "kronos_status": self.status,
            "kronos_evidence": self.markdown,
            "kronos_artifact_dir": self.artifact_dir,
            "kronos_attach_to_market_analyst": self.attach,
        }


def run_kronos_evidence(
    *,
    ticker: str,
    trade_date: str,
    config: dict,
    asset_type: str = "stock",
    now: datetime | None = None,
    bars_loader=None,
    predictor_factory=None,
) -> KronosEvidence:
    """Run Kronos when enabled. Disabled returns immediately and writes nothing."""
    if not config.get("kronos_enabled"):
        return KronosEvidence("disabled", "", "", False)
    try:
        return _run(
            ticker=ticker,
            trade_date=trade_date,
            config=config,
            asset_type=asset_type or "stock",
            now=_clock(now),
            bars_loader=bars_loader,
            predictor_factory=predictor_factory,
        )
    except Exception as exc:
        logger.warning("Kronos evidence failed open: %s", exc, exc_info=True)
        payload = _base_payload(ticker, trade_date, config, _clock(now))
        payload.update(status="failed", reason="unexpected_error", detail=_safe_detail(exc), horizons={})
        return _finish(payload, config, attach=False)


def _run(*, ticker, trade_date, config, asset_type, now, bars_loader, predictor_factory) -> KronosEvidence:
    payload = _base_payload(ticker, trade_date, config, now)
    influence = config.get("kronos_influence") or INFLUENCE_SHADOW
    payload["influence"] = influence
    if influence != INFLUENCE_SHADOW:
        payload.update(
            status="unavailable",
            reason="decision_influence_not_enabled",
            detail=(
                "Only shadow mode is implemented. Kronos does not change ratings, "
                "stops, or size. Leave kronos_influence at 'shadow'."
            ),
        )
        return _finish(payload, config, attach=False)

    if asset_type != "stock":
        payload.update(
            status="unavailable",
            reason="unsupported_asset_type",
            detail=(
                f"Kronos shadow evidence is implemented for stock sessions only, not {asset_type!r}. "
                "A crypto or other session was not guessed."
            ),
        )
        return _finish(payload, config, attach=False)

    try:
        calendar = require_calendar(
            config.get("kronos_calendar") or CALENDAR_ID,
            config.get("kronos_timezone") or DEFAULT_TIMEZONE,
        )
        horizons = _parse_horizons(config.get("kronos_horizons", DEFAULT_HORIZONS))
        lookback = _positive(config.get("kronos_lookback", DEFAULT_LOOKBACK), "kronos_lookback")
        min_bars = _positive(config.get("kronos_min_bars", DEFAULT_MIN_BARS), "kronos_min_bars")
        max_context = _positive(config.get("kronos_max_context", MAX_CONTEXT), "kronos_max_context")
        path_count = _positive(config.get("kronos_path_count", DEFAULT_PATH_COUNT), "kronos_path_count")
        seed = int(config.get("kronos_seed", DEFAULT_SEED))
        temperature = float(config.get("kronos_temperature", DEFAULT_TEMPERATURE))
        top_p = float(config.get("kronos_top_p", DEFAULT_TOP_P))
        datetime.strptime(trade_date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        payload.update(status="unavailable", reason="bad_config", detail=str(exc))
        return _finish(payload, config, attach=False)

    if path_count > MAX_PATH_COUNT:
        payload.update(
            status="unavailable",
            reason="path_count_out_of_range",
            detail=f"kronos_path_count must be 1..{MAX_PATH_COUNT}, got {path_count}",
        )
        return _finish(payload, config, attach=False)
    if not 0 < top_p <= 1:
        payload.update(status="unavailable", reason="bad_config", detail="kronos_top_p must be in (0, 1]")
        return _finish(payload, config, attach=False)

    if lookback < min_bars:
        payload.update(
            status="unavailable",
            reason="bad_config",
            detail=f"kronos_lookback ({lookback}) is below kronos_min_bars ({min_bars})",
        )
        return _finish(payload, config, attach=False)
    if lookback > max_context:
        payload["notes"].append(
            f"lookback {lookback} exceeds max_context {max_context}; the extra history is not sent"
        )
        lookback = max_context

    payload["timezone"] = calendar.timezone
    payload["calendar_id"] = calendar.calendar_id
    payload["calendar_limitation"] = CALENDAR_LIMITATION
    asof = calendar.analysis_asof(trade_date, now)
    payload["asof"] = asof.isoformat()
    payload["horizons_requested"] = horizons
    payload["lookback_requested"] = lookback
    payload["path_count"] = path_count
    payload["seed"] = seed

    try:
        home = resolve_kronos_home(config.get("kronos_home"))
    except KronosUpstreamError as exc:
        payload.update(status="unavailable", reason=exc.reason, detail=exc.detail)
        return _finish(payload, config, attach=False)

    commit = upstream_commit(home)
    warning = commit_warning(commit)
    if warning:
        payload["notes"].append(warning)
    payload["model"] = _model_record(config, home, commit, max_context)

    try:
        loader = bars_loader or _default_bars_loader()
        bars, bar_meta = loader(ticker, trade_date, now=now)
    except Exception as exc:
        logger.warning("Kronos bars unavailable: %s", exc)
        payload.update(status="unavailable", reason="market_data_unavailable", detail=_safe_detail(exc))
        return _finish(payload, config, attach=False)

    payload["adjustment"] = (bar_meta or {}).get("adjustment") or {
        "policy": "unknown",
        "consistent": False,
        "note": "The bar source did not describe corporate-action adjustment.",
    }
    payload["bar_source"] = {key: value for key, value in (bar_meta or {}).items() if key != "adjustment"}

    try:
        history, history_info = frames.completed_history(
            bars, calendar, asof, lookback=lookback, min_bars=min_bars,
        )
    except ValueError as exc:
        payload.update(status="unavailable", reason="bad_bars", detail=str(exc))
        return _finish(payload, config, attach=False)

    payload["history"] = history_info
    payload["notes"].extend(history_info.get("notes") or [])
    if history_info.get("rejection"):
        payload.update(
            status="unavailable",
            reason=history_info["rejection"],
            detail=(
                f"{history_info['rows_completed']} completed bars at {asof.isoformat()}; "
                f"kronos_min_bars is {min_bars}. No forecast was filled in."
            ),
        )
        return _finish(payload, config, attach=_attach(config), files=_observed_only(history))

    try:
        model_frame, amount_meta = frames.model_inputs(
            history, config.get("kronos_amount_policy") or AMOUNT_POLICY_OMIT,
        )
    except ValueError as exc:
        payload.update(status="unavailable", reason="bad_config", detail=str(exc))
        return _finish(payload, config, attach=False, files=_observed_only(history))

    payload["amount"] = amount_meta
    payload["notes"].extend(amount_meta.get("notes") or [])
    payload["output_conditioning"] = _conditioning(amount_meta)
    stamps = frames.history_stamps(history["session_date"], calendar)
    last_session = history["session_date"].iloc[-1]
    observed = _observed_frame(history)
    model_csv = _model_csv(history, stamps, model_frame)

    factory = predictor_factory or _default_predictor_factory(config, max_context)
    try:
        predictor = factory(home)
    except KronosUpstreamError as exc:
        payload.update(status="unavailable", reason=exc.reason, detail=exc.detail)
        return _finish(payload, config, attach=_attach(config), files={
            "observed_ohlcv.csv": observed, "model_input.csv": model_csv,
        })
    except Exception as exc:
        logger.warning("Kronos predictor unavailable: %s", exc)
        payload.update(status="unavailable", reason="predictor_unavailable", detail=_safe_detail(exc))
        return _finish(payload, config, attach=_attach(config), files={
            "observed_ohlcv.csv": observed, "model_input.csv": model_csv,
        })

    horizon_payloads = {}
    files = {"observed_ohlcv.csv": observed, "model_input.csv": model_csv}
    any_failed = False
    for horizon in horizons:
        sessions, future = frames.future_stamps(last_session, horizon, calendar)
        try:
            result = predict.forecast_horizon(
                predictor, model_frame, stamps, future,
                horizon=horizon, base_seed=seed, path_count=path_count,
                temperature=temperature, top_p=top_p,
            )
        except Exception as exc:
            logger.warning("Kronos horizon %s failed: %s", horizon, exc)
            any_failed = True
            horizon_payloads[str(horizon)] = {
                "status": "failed",
                "horizon": horizon,
                "detail": _safe_detail(exc),
                "sample_count_passed_to_predict": 1,
                "rows": None,
            }
            continue
        public = _public_horizon(result, sessions)
        public["conditioning"] = payload["output_conditioning"]
        horizon_payloads[str(horizon)] = public
        if path_count > 1:
            files[f"paths_h{horizon}.csv"] = _paths_frame(result["paths"], public["sessions"])

    payload["horizons"] = horizon_payloads
    if any_failed:
        payload.update(
            status="failed",
            reason="inference_failed",
            detail="One or more horizons did not return a finite path. Horizons that did are included; the others are empty.",
        )
    else:
        payload.update(status="ok", reason=None, detail="Shadow forecast only. No rating, size, or stop was changed.")

    forecast_csv = artifacts.forecast_table(horizon_payloads)
    files["forecast.csv"] = forecast_csv if not forecast_csv.empty else None
    return _finish(payload, config, attach=_attach(config), files=files)


def _finish(payload: dict, config: dict, *, attach: bool, files: dict | None = None) -> KronosEvidence:
    markdown = artifacts.render_markdown(payload)
    payload["markdown"] = markdown
    directory = ""
    try:
        path = _artifact_path(config, payload.get("ticker") or "", payload.get("trade_date") or "")
    except ValueError as exc:
        logger.warning("Kronos artifacts skipped: %s", exc)
        path = None
    if path is not None:
        try:
            artifacts.write_run_dir(path, payload, files or {})
            directory = str(path)
        except Exception as exc:
            logger.warning("Kronos artifacts could not be written: %s", exc)
            markdown += f"\n\nArtifacts could not be written ({type(exc).__name__}).\n"
    # Attachment is how the market analyst sees the text. Shadow default leaves it off.
    # A refused decision-influence mode never attaches.
    do_attach = bool(attach and payload.get("influence") == INFLUENCE_SHADOW and payload.get("status") != "disabled")
    return KronosEvidence(payload["status"], markdown, directory, do_attach)


def _base_payload(ticker: str, trade_date: str, config: dict, now: datetime) -> dict:
    return {
        "status": "unavailable",
        "reason": None,
        "detail": "",
        "ticker": ticker,
        "trade_date": trade_date,
        "generated_at": now.astimezone(UTC).isoformat(),
        "influence": INFLUENCE_SHADOW,
        "decision_influence": False,
        "rating_changed": False,
        "stops_changed": False,
        "sizing_changed": False,
        "confidence_adjusted": False,
        "pretraining_cutoff": PRETRAINING_CUTOFF,
        "pretraining_contamination": PRETRAINING_CONTAMINATION,
        "pretraining_note": PRETRAINING_NOTE,
        "paper": PAPER_URL,
        "license": LICENSE_NAME,
        "license_notice": LICENSE_NOTICE,
        "upstream_repo": KRONOS_REPO,
        "reviewed_upstream_commit": REVIEWED_UPSTREAM_COMMIT,
        "notes": [],
        "horizons": {},
        "device": config.get("kronos_device") or DEFAULT_DEVICE,
    }


def _model_record(config: dict, home, commit: str | None, max_context: int) -> dict:
    return {
        "id": config.get("kronos_model_id") or MODEL_ID,
        "revision": config.get("kronos_model_revision") or MODEL_REVISION,
        "tokenizer_id": config.get("kronos_tokenizer_id") or TOKENIZER_ID,
        "tokenizer_revision": config.get("kronos_tokenizer_revision") or TOKENIZER_REVISION,
        "max_context": max_context,
        "device": config.get("kronos_device") or DEFAULT_DEVICE,
        "upstream_commit": commit,
        "reviewed_upstream_commit": REVIEWED_UPSTREAM_COMMIT,
        "license": LICENSE_NAME,
        "weights_bundled": False,
    }


def _default_predictor_factory(config: dict, max_context: int):
    def factory(home):
        return load_predictor(
            home,
            model_id=config.get("kronos_model_id") or MODEL_ID,
            model_revision=config.get("kronos_model_revision") or MODEL_REVISION,
            tokenizer_id=config.get("kronos_tokenizer_id") or TOKENIZER_ID,
            tokenizer_revision=config.get("kronos_tokenizer_revision") or TOKENIZER_REVISION,
            device=config.get("kronos_device") or DEFAULT_DEVICE,
            max_context=max_context,
        )
    return factory


def _default_bars_loader():
    from tradingagents.dataflows.vendors.yahoo.kronos_bars import load_split_consistent_daily_bars
    return load_split_consistent_daily_bars


def _public_horizon(result: dict, sessions: list) -> dict:
    session_text = [day.isoformat() for day in sessions]
    public = {
        key: value for key, value in result.items() if key not in {"mean", "dispersion", "paths", "sampling"}
    }
    public["sampling"] = result["sampling"]
    public["sessions"] = session_text
    public["rows"] = artifacts.rows_from_frame(result["mean"], session_text)
    public["dispersion_rows"] = (
        artifacts.rows_from_frame(result["dispersion"], session_text) if result["dispersion"] is not None else None
    )
    public["status"] = "ok"
    return public


def _paths_frame(paths, sessions: list[str]):
    import pandas as pd

    records = []
    for index, path in enumerate(paths):
        for session, (_, row) in zip(sessions, path.iterrows(), strict=True):
            item = {"path": index, "session": session}
            for column in path.columns:
                item[column] = float(row[column])
            records.append(item)
    return pd.DataFrame.from_records(records)


def _observed_frame(history):
    frame = history.copy()
    frame["session_date"] = frame["session_date"].map(lambda day: day.isoformat())
    return frame


def _model_csv(history, stamps, model_frame):
    frame = model_frame.copy()
    frame.insert(0, "timestamp", [stamp.isoformat() for stamp in stamps])
    frame.insert(0, "session_date", [day.isoformat() for day in history["session_date"]])
    return frame


def _observed_only(history) -> dict:
    if history is None or len(history) == 0:
        return {}
    return {"observed_ohlcv.csv": _observed_frame(history)}


def _conditioning(amount_meta: dict) -> str:
    if amount_meta.get("amount_source") == "observed":
        return "conditioned_on_observed_amount"
    if amount_meta.get("amount_is_approximation"):
        return "conditioned_on_labeled_amount_approximation"
    return "volume_and_amount_outputs_follow_zeros_filled_for_omitted_inputs"


def _artifact_path(config: dict, ticker: str, trade_date: str) -> Path:
    safe = safe_ticker_component(ticker)
    if not trade_date or "/" in trade_date or "\\" in trade_date:
        raise ValueError("trade_date cannot be a path")
    root = Path(config.get("results_dir") or ".")
    return root / "kronos" / f"{safe}_{trade_date}"


def _attach(config: dict) -> bool:
    return bool(config.get("kronos_attach_to_market_analyst"))


def _parse_horizons(value) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    horizons = [int(part) for part in parts]
    if not horizons or any(item < 1 for item in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("kronos_horizons must be unique positive integers, for example 5,20")
    return horizons


def _positive(value, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def _clock(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now


def _safe_detail(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:500]
