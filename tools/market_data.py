from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
UNIVERSE_PATH = PROJECT_ROOT / "config" / "universe.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _settings() -> dict[str, Any]:
    return _read_yaml(SETTINGS_PATH)


def _akshare_date(value: str | date | datetime | None) -> str:
    if value is None:
        return date.today().strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return str(value).replace("-", "")


def load_watchlist() -> list[dict[str, Any]]:
    """Load stock symbols from config/universe.yaml."""
    universe = _read_yaml(UNIVERSE_PATH)
    watchlist = universe.get("watchlist", [])

    if not isinstance(watchlist, list):
        raise ValueError(f"watchlist must be a list in {UNIVERSE_PATH}")

    normalized: list[dict[str, Any]] = []
    for item in watchlist:
        if isinstance(item, str):
            normalized.append({"symbol": item})
            continue

        if not isinstance(item, dict) or not item.get("symbol"):
            raise ValueError(f"invalid watchlist item: {item!r}")

        normalized.append({**item, "symbol": str(item["symbol"]).zfill(6)})

    return normalized


def fetch_stock_daily(symbol: str, start: str | date | datetime, end: str | date | datetime | None = None) -> pd.DataFrame:
    """Fetch daily A-share bars from AKShare and normalize column names."""
    import akshare as ak

    data_settings = _settings().get("data", {})
    adjust = data_settings.get("adjust", "qfq")

    raw = ak.stock_zh_a_hist(
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=_akshare_date(start),
        end_date=_akshare_date(end),
        adjust=adjust,
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "change",
                "turnover",
            ]
        )

    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = raw.rename(columns=column_map)

    expected_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"AKShare response missing columns for {symbol}: {missing}")

    df = df[expected_columns].copy()
    df.insert(1, "symbol", str(symbol).zfill(6))
    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [column for column in df.columns if column not in {"date", "symbol"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


def save_daily_to_parquet(df: pd.DataFrame, symbol: str) -> Path:
    """Save one symbol's daily bars into data/raw/{symbol}.parquet."""
    data_settings = _settings().get("data", {})
    raw_dir = PROJECT_ROOT / data_settings.get("raw_dir", "data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / f"{str(symbol).zfill(6)}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path
