from __future__ import annotations

import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _prepare_price_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    _require_columns(df, columns)

    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        if result["date"].isna().any():
            raise ValueError("date column contains invalid values")
        result = result.sort_values("date").reset_index(drop=True)
    return result


def calc_ma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")

    result = _prepare_price_frame(df, ["close"])
    close = pd.to_numeric(result["close"], errors="coerce")
    result[f"ma{window}"] = close.rolling(window=window, min_periods=window).mean()
    return result


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")

    result = _prepare_price_frame(df, ["high", "low", "close"])
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    close = pd.to_numeric(result["close"], errors="coerce")
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result[f"atr{window}"] = true_range.rolling(window=window, min_periods=window).mean()
    return result


def calc_amount_ratio(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")

    result = _prepare_price_frame(df, ["amount"])
    amount = pd.to_numeric(result["amount"], errors="coerce")
    previous_average = amount.shift(1).rolling(window=window, min_periods=window).mean()
    previous_average = previous_average.mask(previous_average == 0)
    result[f"amount_ratio_{window}d"] = amount.div(previous_average)
    return result


def calc_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    result = _prepare_price_frame(df, ["close"])
    close = pd.to_numeric(result["close"], errors="coerce")
    result["pct_chg"] = close.pct_change()
    return result


def calc_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = calc_ma(df, 5)
    result = calc_ma(result, 10)
    result = calc_ma(result, 20)
    result = calc_atr(result, 14)
    result = calc_amount_ratio(result, 5)
    result = calc_pct_change(result)
    return result


def calc_amount_change(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    return calc_amount_ratio(df, window)
