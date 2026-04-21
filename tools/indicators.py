from __future__ import annotations

import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def calc_ma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate close-price moving average and append ma{window}."""
    if window <= 0:
        raise ValueError("window must be positive")

    _require_columns(df, ["close"])

    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    result[f"ma{window}"] = close.rolling(window=window, min_periods=window).mean()
    return result


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Average True Range and append atr{window}."""
    if window <= 0:
        raise ValueError("window must be positive")

    _require_columns(df, ["high", "low", "close"])

    result = df.copy()
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


def calc_amount_change(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calculate current amount divided by previous-window average amount."""
    if window <= 0:
        raise ValueError("window must be positive")

    _require_columns(df, ["amount"])

    result = df.copy()
    amount = pd.to_numeric(result["amount"], errors="coerce")
    previous_average = amount.shift(1).rolling(window=window, min_periods=window).mean()
    result[f"amount_ratio_{window}d"] = amount / previous_average
    return result


def calc_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Day4 review indicator set."""
    result = calc_ma(df, 5)
    result = calc_ma(result, 10)
    result = calc_ma(result, 20)
    result = calc_atr(result, 14)
    result = calc_amount_change(result, 5)
    return result
