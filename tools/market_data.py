from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
UNIVERSE_PATH = CONFIG_DIR / "universe.yaml"
POSITIONS_PATH = CONFIG_DIR / "positions.yaml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
STANDARD_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]

_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "time": "date",
    "trade_date": "date",
    "tradedate": "date",
    "日期": "date",
    "交易日期": "date",
    "open": "open",
    "open_price": "open",
    "openprice": "open",
    "开盘": "open",
    "high": "high",
    "high_price": "high",
    "highprice": "high",
    "最高": "high",
    "low": "low",
    "low_price": "low",
    "lowprice": "low",
    "最低": "low",
    "close": "close",
    "close_price": "close",
    "closeprice": "close",
    "收盘": "close",
    "volume": "volume",
    "vol": "volume",
    "成交量": "volume",
    "成交量股": "volume",
    "成交量手": "volume",
    "amount": "amount",
    "成交额": "amount",
    "成交金额": "amount",
    "turnover": "turnover",
    "turnover_rate": "turnover",
    "换手率": "turnover",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _settings() -> dict[str, Any]:
    return _read_yaml(SETTINGS_PATH)


@contextmanager
def _proxy_context(use_system_proxy: bool = True):
    if use_system_proxy:
        yield
        return

    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]
    original = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            os.environ.pop(key, None)
        # Tell requests/urllib to bypass any OS-discovered proxy settings.
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _raw_data_dir() -> Path:
    data_settings = _settings().get("data", {})
    raw_dir = data_settings.get("raw_dir", "data/raw")
    return (PROJECT_ROOT / raw_dir).resolve()


def _normalize_text(value: Any) -> str:
    text = str(value).strip().replace("\ufeff", "")
    replacements = {
        " ": "",
        "\t": "",
        "\n": "",
        "\r": "",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "-": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.casefold()


def _normalize_symbol(symbol: Any) -> str:
    text = str(symbol).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits.zfill(6)
    return text.zfill(6)


def _normalize_watchlist_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        symbol = _normalize_symbol(item)
        return {"symbol": symbol, "name": symbol}

    if not isinstance(item, dict) or not item.get("symbol"):
        raise ValueError(f"invalid watchlist item: {item!r}")

    normalized = dict(item)
    symbol = _normalize_symbol(item["symbol"])
    normalized["symbol"] = symbol
    normalized["name"] = str(item.get("name") or symbol)
    return normalized


def _normalize_position_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or not item.get("symbol"):
        raise ValueError(f"invalid position item: {item!r}")

    normalized = dict(item)
    normalized["symbol"] = _normalize_symbol(item["symbol"])
    if "name" in normalized and normalized["name"] is not None:
        normalized["name"] = str(normalized["name"])
    for key in ("cost", "shares"):
        if key in normalized and normalized[key] is not None:
            value = pd.to_numeric(pd.Series([normalized[key]]), errors="coerce").iloc[0]
            normalized[key] = value.item() if hasattr(value, "item") else value
    return normalized


def _standardize_column_name(column: Any) -> str:
    normalized = _normalize_text(column)
    normalized = normalized.replace("(", "").replace(")", "")
    normalized = normalized.replace("[", "").replace("]", "")
    return _COLUMN_ALIASES.get(normalized, normalized)


def load_universe() -> dict[str, Any]:
    universe = _read_yaml(UNIVERSE_PATH)
    watchlist = universe.get("watchlist", [])
    focus_list = universe.get("focus_list", [])

    if not isinstance(watchlist, list):
        raise ValueError(f"watchlist must be a list in {UNIVERSE_PATH}")
    if focus_list is None:
        focus_list = []
    if not isinstance(focus_list, list):
        raise ValueError(f"focus_list must be a list in {UNIVERSE_PATH}")

    normalized_watchlist = [_normalize_watchlist_item(item) for item in watchlist]
    normalized_focus = [_normalize_symbol(item) for item in focus_list]
    return {
        **universe,
        "watchlist": normalized_watchlist,
        "focus_list": normalized_focus,
    }


def load_watchlist() -> list[dict[str, Any]]:
    return load_universe()["watchlist"]


def load_positions() -> list[dict[str, Any]]:
    cfg = _read_yaml(POSITIONS_PATH)
    positions = cfg.get("positions", [])
    if not isinstance(positions, list):
        raise ValueError(f"positions must be a list in {POSITIONS_PATH}")
    return [_normalize_position_item(item) for item in positions]


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STANDARD_OHLCV_COLUMNS)

    renamed = df.rename(columns={column: _standardize_column_name(column) for column in df.columns}).copy()

    duplicated = renamed.columns[renamed.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"duplicate columns after normalization: {duplicated}")

    missing = [column for column in STANDARD_OHLCV_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"missing required columns after normalization: {missing}")

    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
    if renamed["date"].isna().any():
        raise ValueError("date column contains invalid values")

    for column in STANDARD_OHLCV_COLUMNS[1:]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

    standardized = renamed[STANDARD_OHLCV_COLUMNS].sort_values("date").reset_index(drop=True)
    return standardized


def load_symbol_df(symbol: str, data_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(data_dir) if data_dir is not None else _raw_data_dir()
    normalized_symbol = _normalize_symbol(symbol)
    parquet_path = raw_dir / f"{normalized_symbol}.parquet"
    csv_path = raw_dir / f"{normalized_symbol}.csv"

    if parquet_path.exists():
        raw_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        raw_df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"market data file not found for {normalized_symbol}: {parquet_path} or {csv_path}")

    standardized_source = raw_df.rename(columns={column: _standardize_column_name(column) for column in raw_df.columns}).copy()
    standardized_source["date"] = pd.to_datetime(standardized_source["date"], errors="coerce")
    standardized_source = standardized_source.sort_values("date").reset_index(drop=True)

    normalized_df = normalize_ohlcv_columns(raw_df)

    if "symbol" in standardized_source.columns:
        normalized_df.insert(1, "symbol", standardized_source["symbol"].astype(str).map(_normalize_symbol))
    else:
        normalized_df.insert(1, "symbol", normalized_symbol)

    extra_columns = [column for column in standardized_source.columns if column not in normalized_df.columns]
    if extra_columns:
        normalized_df = pd.concat([normalized_df, standardized_source[extra_columns]], axis=1)

    return normalized_df


def _akshare_date(value: str | date | datetime | None) -> str:
    if value is None:
        return date.today().strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return str(value).replace("-", "")


def _eastmoney_market_id(symbol: str) -> str:
    normalized_symbol = _normalize_symbol(symbol)
    if normalized_symbol.startswith(("5", "6", "9")):
        return "1"
    return "0"


def _sina_symbol(symbol: str) -> str:
    normalized_symbol = _normalize_symbol(symbol)
    if normalized_symbol.startswith(("4", "8")):
        return f"bj{normalized_symbol}"
    if normalized_symbol.startswith(("5", "6", "9")):
        return f"sh{normalized_symbol}"
    return f"sz{normalized_symbol}"


def _fetch_stock_daily_from_sina(
    symbol: str,
    start: str | date | datetime,
    end: str | date | datetime | None,
    adjust: str,
    use_system_proxy: bool,
) -> pd.DataFrame:
    import akshare as ak

    with _proxy_context(use_system_proxy=use_system_proxy):
        raw = ak.stock_zh_a_daily(
            symbol=_sina_symbol(symbol),
            start_date=_akshare_date(start),
            end_date=_akshare_date(end),
            adjust=adjust,
        )

    if raw.empty:
        return pd.DataFrame()

    df = raw.rename(columns={column: _standardize_column_name(column) for column in raw.columns})
    return df


def _fetch_stock_daily_from_eastmoney(
    symbol: str,
    start: str | date | datetime,
    end: str | date | datetime | None,
    adjust: str,
    timeout: float,
) -> pd.DataFrame:
    adjust_dict = {"": "0", "qfq": "1", "hfq": "2"}
    period_dict = {"daily": "101"}
    normalized_symbol = _normalize_symbol(symbol)
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_dict["daily"],
        "fqt": adjust_dict.get(adjust, "1"),
        "secid": f"{_eastmoney_market_id(normalized_symbol)}.{normalized_symbol}",
        "beg": _akshare_date(start),
        "end": _akshare_date(end),
        "_": "1623766962675",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
    }

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    klines = payload.get("data", {}).get("klines") or []
    if not klines:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 11:
            raise ValueError(f"unexpected Eastmoney kline payload for {normalized_symbol}: {item!r}")
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
                "amplitude": parts[7],
                "pct_change": parts[8],
                "change": parts[9],
                "turnover": parts[10],
            }
        )

    return pd.DataFrame(rows)


def _fetch_stock_daily_raw(
    symbol: str,
    start: str | date | datetime,
    end: str | date | datetime | None,
    adjust: str,
    use_system_proxy: bool,
) -> pd.DataFrame:
    import akshare as ak

    with _proxy_context(use_system_proxy=use_system_proxy):
        return ak.stock_zh_a_hist(
            symbol=_normalize_symbol(symbol),
            period="daily",
            start_date=_akshare_date(start),
            end_date=_akshare_date(end),
            adjust=adjust,
        )


def _proxy_modes(preferred_use_system_proxy: bool) -> list[bool]:
    modes = [preferred_use_system_proxy]
    alternate = not preferred_use_system_proxy
    if alternate not in modes:
        modes.append(alternate)
    return modes


def fetch_stock_daily(symbol: str, start: str | date | datetime, end: str | date | datetime | None = None) -> pd.DataFrame:
    data_settings = _settings().get("data", {})
    adjust = data_settings.get("adjust", "qfq")
    use_system_proxy = data_settings.get("use_system_proxy", False)
    retry_count = max(int(data_settings.get("retry_count", 3)), 1)
    retry_delay = max(float(data_settings.get("retry_delay_seconds", 1.5)), 0.0)
    request_timeout = max(float(data_settings.get("request_timeout_seconds", 15.0)), 1.0)

    primary_errors: list[str] = []
    sina_errors: list[str] = []
    fallback_errors: list[str] = []
    raw = pd.DataFrame()
    for proxy_mode in _proxy_modes(use_system_proxy):
        for attempt in range(1, retry_count + 1):
            try:
                raw = _fetch_stock_daily_raw(
                    symbol=symbol,
                    start=start,
                    end=end,
                    adjust=adjust,
                    use_system_proxy=proxy_mode,
                )
                primary_errors.clear()
                break
            except Exception as exc:
                primary_errors.append(
                    f"AKShare attempt {attempt}/{retry_count} failed for {_normalize_symbol(symbol)} "
                    f"(use_system_proxy={proxy_mode}): {exc}"
                )
                if attempt >= retry_count:
                    break
                time.sleep(retry_delay * attempt)
        if not raw.empty:
            break

    if raw.empty:
        for proxy_mode in _proxy_modes(use_system_proxy):
            for attempt in range(1, retry_count + 1):
                try:
                    raw = _fetch_stock_daily_from_sina(
                        symbol=symbol,
                        start=start,
                        end=end,
                        adjust=adjust,
                        use_system_proxy=proxy_mode,
                    )
                    sina_errors.clear()
                    break
                except Exception as exc:
                    sina_errors.append(
                        f"Sina fallback attempt {attempt}/{retry_count} failed for {_normalize_symbol(symbol)} "
                        f"(use_system_proxy={proxy_mode}): {exc}"
                    )
                    if attempt >= retry_count:
                        break
                    time.sleep(retry_delay * attempt)
            if not raw.empty:
                break

    if raw.empty:
        for proxy_mode in _proxy_modes(use_system_proxy):
            for attempt in range(1, retry_count + 1):
                try:
                    with _proxy_context(use_system_proxy=proxy_mode):
                        raw = _fetch_stock_daily_from_eastmoney(
                            symbol=symbol,
                            start=start,
                            end=end,
                            adjust=adjust,
                            timeout=request_timeout,
                        )
                    fallback_errors.clear()
                    break
                except Exception as exc:
                    fallback_errors.append(
                        f"Eastmoney fallback attempt {attempt}/{retry_count} failed for {_normalize_symbol(symbol)} "
                        f"(use_system_proxy={proxy_mode}): {exc}"
                    )
                    if attempt >= retry_count:
                        break
                    time.sleep(retry_delay * attempt)
            if not raw.empty:
                break

    error_messages = [errors[-1] for errors in (primary_errors, sina_errors, fallback_errors) if errors]

    if raw.empty and error_messages:
        raise RuntimeError("; ".join(error_messages))

    if raw.empty:
        return pd.DataFrame(columns=["date", "symbol", *STANDARD_OHLCV_COLUMNS[1:], "amplitude", "pct_change", "change", "turnover"])

    df = raw.rename(columns={column: _standardize_column_name(column) for column in raw.columns})
    df = normalize_ohlcv_columns(df)
    df.insert(1, "symbol", _normalize_symbol(symbol))

    optional_numeric_columns = {
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    for raw_name, target_name in optional_numeric_columns.items():
        if raw_name in raw.columns:
            df[target_name] = pd.to_numeric(raw[raw_name], errors="coerce")

    return df


def save_daily_to_parquet(df: pd.DataFrame, symbol: str) -> Path:
    raw_dir = _raw_data_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    standardized_source = df.rename(columns={column: _standardize_column_name(column) for column in df.columns}).copy()
    if "date" in standardized_source.columns:
        standardized_source["date"] = pd.to_datetime(standardized_source["date"], errors="coerce")
        standardized_source = standardized_source.sort_values("date").reset_index(drop=True)

    normalized_df = normalize_ohlcv_columns(df)
    if "symbol" in standardized_source.columns:
        normalized_df.insert(1, "symbol", standardized_source["symbol"].astype(str).map(_normalize_symbol))
    else:
        normalized_df.insert(1, "symbol", _normalize_symbol(symbol))
    output_path = raw_dir / f"{_normalize_symbol(symbol)}.parquet"
    normalized_df.to_parquet(output_path, index=False)
    return output_path
