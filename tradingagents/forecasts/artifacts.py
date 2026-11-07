"""Audit files for one Kronos run.

Observed bars, the exact model input, and the forecast live in different
files. A failed or unavailable run writes the manifest and the markdown and
does not write a numeric forecast.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.files import replace_file


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temp: Path(temp).write_text(text, encoding="utf-8"))


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(temp: str) -> None:
        frame.to_csv(temp, index=False)

    replace_file(path, _write)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def render_markdown(payload: dict) -> str:
    """Human-readable shadow evidence. JSON and CSV remain the full-precision record."""
    status = payload.get("status")
    lines = [
        "# Kronos shadow evidence",
        "",
        f"- Status: {status}",
        f"- Reason: {payload.get('reason') or 'none'}",
        "- Influence: shadow. This is not a rating, a position size, or a stop.",
        "- It does not replace the market analyst, the debate, or the risk team.",
        "- Decision influence: off. No automatic sizing, stop, or rating change is applied.",
        "- Confidence is not adjusted when a path agrees with indicators from the same prices.",
        "",
        payload.get("detail") or "",
        "",
        f"- Pretraining contamination: {payload.get('pretraining_contamination')}",
        f"- {payload.get('pretraining_note')}",
        "",
    ]
    if payload.get("calendar_limitation"):
        lines.append(f"- Calendar: {payload['calendar_limitation']}")
        lines.append("")
    horizons = payload.get("horizons") or {}
    if not horizons:
        lines.append("No numeric forecast was produced.")
        lines.append("")
    for name, horizon in horizons.items():
        lines.extend(_horizon_markdown(name, horizon))
    notes = payload.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    lines.append(
        "Observed bars are in `observed_ohlcv.csv`. The model input is in "
        "`model_input.csv`. Forecast numbers, when status is ok for a horizon, "
        "are in `forecast.json` and `forecast.csv`. Those files are not observations."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _horizon_markdown(name: str, horizon: dict) -> list[str]:
    lines = [f"## Horizon {name} trading sessions", ""]
    if horizon.get("status") != "ok":
        lines.append(f"Not produced ({horizon.get('status')}): {horizon.get('detail') or ''}")
        lines.append("")
        return lines
    lines.append(f"- Sessions (forecast, not observed): {', '.join(horizon.get('sessions') or [])}")
    lines.append(f"- Seeds: {horizon.get('seeds')}")
    lines.append(f"- sample_count passed to predict(): {horizon.get('sample_count_passed_to_predict')}")
    lines.append(f"- Aggregation: {horizon.get('aggregation')}")
    lines.append(f"- {horizon.get('dispersion_note')}")
    lines.append("")
    lines.extend(_table(horizon.get("rows") or []))
    if horizon.get("dispersion_rows"):
        lines.append("")
        lines.append("Empirical dispersion (sample standard deviation across paths, not a probability):")
        lines.append("")
        lines.extend(_table(horizon["dispersion_rows"]))
    lines.append("")
    return lines


def _table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = [_cell(row.get(column)) for column in columns]
        body.append("| " + " | ".join(cells) + " |")
    return [header, rule, *body]


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def rows_from_frame(frame: pd.DataFrame, sessions: list[str]) -> list[dict]:
    rows = []
    for session, (_, row) in zip(sessions, frame.iterrows(), strict=True):
        item = {"session": session}
        for column in frame.columns:
            item[column] = float(row[column])
        rows.append(item)
    return rows


def forecast_table(horizons: dict) -> pd.DataFrame:
    """Long forecast table. One block per successful horizon. No observed rows."""
    records = []
    for name, horizon in horizons.items():
        if horizon.get("status") != "ok":
            continue
        for kind, rows in (("mean", horizon.get("rows") or []), ("dispersion", horizon.get("dispersion_rows") or [])):
            for row in rows:
                records.append({"horizon": int(name), "series": kind, **row})
    return pd.DataFrame.from_records(records)


def write_run_dir(directory: Path, payload: dict, files: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Write frames and the manifest. ``files`` values of None are skipped."""
    directory.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, frame in files.items():
        if frame is None:
            continue
        path = directory / name
        write_frame(path, frame)
        hashes[name] = sha256_file(path)
    markdown = render_markdown(payload)
    md_path = directory / "forecast.md"
    replace_file(md_path, lambda temp: Path(temp).write_text(markdown, encoding="utf-8"))
    hashes["forecast.md"] = sha256_file(md_path)
    forecast_json = {key: value for key, value in payload.items() if key != "markdown"}
    write_json(directory / "forecast.json", forecast_json)
    hashes["forecast.json"] = sha256_file(directory / "forecast.json")
    manifest = {
        "artifact_dir": str(directory),
        "files": hashes,
        **{key: payload[key] for key in (
            "status", "reason", "ticker", "trade_date", "asof", "timezone",
            "calendar_id", "influence", "decision_influence",
        ) if key in payload},
    }
    write_json(directory / "manifest.json", manifest)
    return hashes
