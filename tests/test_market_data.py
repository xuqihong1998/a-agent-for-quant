from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from tools.market_data import (
    STANDARD_OHLCV_COLUMNS,
    _proxy_context,
    fetch_stock_daily,
    load_positions,
    load_symbol_df,
    load_universe,
    normalize_ohlcv_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class MarketDataTest(unittest.TestCase):
    def test_proxy_context_disables_and_restores_proxy_env(self) -> None:
        original_https_proxy = os.environ.get("HTTPS_PROXY")
        original_no_proxy = os.environ.get("NO_PROXY")
        try:
            os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
            os.environ.pop("NO_PROXY", None)

            with _proxy_context(use_system_proxy=False):
                self.assertNotIn("HTTPS_PROXY", os.environ)
                self.assertEqual(os.environ.get("NO_PROXY"), "*")

            if original_https_proxy is None:
                self.assertNotIn("HTTPS_PROXY", os.environ)
            else:
                self.assertEqual(os.environ.get("HTTPS_PROXY"), original_https_proxy)

            if original_no_proxy is None:
                self.assertNotIn("NO_PROXY", os.environ)
            else:
                self.assertEqual(os.environ.get("NO_PROXY"), original_no_proxy)
        finally:
            if original_https_proxy is None:
                os.environ.pop("HTTPS_PROXY", None)
            else:
                os.environ["HTTPS_PROXY"] = original_https_proxy

            if original_no_proxy is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = original_no_proxy

    def test_normalize_ohlcv_columns_from_chinese_headers(self) -> None:
        df = pd.DataFrame(
            {
                " 日期 ": ["2026-04-18", "2026-04-17"],
                "开盘": ["10.1", "9.8"],
                "最高": ["10.5", "10.2"],
                "最低": ["9.9", "9.7"],
                "收盘": ["10.4", "10.0"],
                "成交量": ["1000", "800"],
                "成交额": ["1000000", "800000"],
            }
        )

        normalized = normalize_ohlcv_columns(df)

        self.assertEqual(normalized.columns.tolist(), STANDARD_OHLCV_COLUMNS)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized["date"]))
        self.assertEqual(normalized["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-04-17", "2026-04-18"])

    def test_load_universe_and_positions_are_normalized(self) -> None:
        universe = load_universe()
        positions = load_positions()

        self.assertIsInstance(universe["watchlist"], list)
        self.assertIn("focus_list", universe)
        self.assertTrue(all("symbol" in item and "name" in item for item in universe["watchlist"]))
        self.assertTrue(all(len(item["symbol"]) == 6 for item in universe["watchlist"]))
        self.assertTrue(all(len(item["symbol"]) == 6 for item in positions))

    def test_load_symbol_df_returns_standard_columns(self) -> None:
        parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            self.skipTest(f"no raw data found in {RAW_DATA_DIR}")

        symbol = parquet_files[0].stem
        df = load_symbol_df(symbol)

        for column in ["date", "symbol", *STANDARD_OHLCV_COLUMNS[1:]]:
            self.assertIn(column, df.columns)
        self.assertTrue(df["date"].is_monotonic_increasing)

    def test_fetch_stock_daily_retries_after_transient_failure(self) -> None:
        attempts = {"count": 0}
        raw_df = pd.DataFrame(
            {
                "日期": ["2026-04-18"],
                "开盘": [10.1],
                "最高": [10.5],
                "最低": [9.9],
                "收盘": [10.4],
                "成交量": [1000],
                "成交额": [1000000],
            }
        )

        def flaky_fetch(**_: object) -> pd.DataFrame:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("Remote end closed connection without response")
            return raw_df

        with patch("tools.market_data._fetch_stock_daily_raw", side_effect=flaky_fetch), patch(
            "tools.market_data._settings",
            return_value={"data": {"retry_count": 3, "retry_delay_seconds": 0}},
        ):
            df = fetch_stock_daily("002074", "2026-04-01", "2026-04-18")

        self.assertEqual(attempts["count"], 3)
        self.assertEqual(df["symbol"].tolist(), ["002074"])
        self.assertEqual(df["close"].tolist(), [10.4])

    def test_fetch_stock_daily_falls_back_to_eastmoney(self) -> None:
        eastmoney_df = pd.DataFrame(
            {
                "date": ["2026-04-18"],
                "open": [10.1],
                "close": [10.4],
                "high": [10.5],
                "low": [9.9],
                "volume": [1000],
                "amount": [1000000],
                "amplitude": [6.06],
                "pct_change": [4.0],
                "change": [0.4],
                "turnover": [1.2],
            }
        )

        with patch(
            "tools.market_data._fetch_stock_daily_raw",
            side_effect=ConnectionError("Remote end closed connection without response"),
        ), patch(
            "tools.market_data._fetch_stock_daily_from_sina",
            side_effect=ConnectionError("Sina host unavailable"),
        ), patch(
            "tools.market_data._fetch_stock_daily_from_eastmoney",
            return_value=eastmoney_df,
        ) as fallback_fetch, patch(
            "tools.market_data._settings",
            return_value={"data": {"retry_count": 2, "retry_delay_seconds": 0, "request_timeout_seconds": 5}},
        ):
            df = fetch_stock_daily("002074", "2026-04-01", "2026-04-18")

        self.assertEqual(fallback_fetch.call_count, 1)
        self.assertEqual(df["symbol"].tolist(), ["002074"])
        self.assertEqual(df["turnover"].tolist(), [1.2])

    def test_fetch_stock_daily_falls_back_to_sina(self) -> None:
        sina_df = pd.DataFrame(
            {
                "date": ["2026-04-18"],
                "open": [10.1],
                "high": [10.5],
                "low": [9.9],
                "close": [10.4],
                "volume": [1000],
                "amount": [1000000],
            }
        )

        with patch(
            "tools.market_data._fetch_stock_daily_raw",
            side_effect=ConnectionError("Eastmoney host unavailable"),
        ), patch(
            "tools.market_data._fetch_stock_daily_from_sina",
            return_value=sina_df,
        ) as sina_fetch, patch(
            "tools.market_data._fetch_stock_daily_from_eastmoney",
            side_effect=AssertionError("Eastmoney direct fallback should not run"),
        ), patch(
            "tools.market_data._settings",
            return_value={"data": {"retry_count": 2, "retry_delay_seconds": 0, "request_timeout_seconds": 5}},
        ):
            df = fetch_stock_daily("002074", "2026-04-01", "2026-04-18")

        self.assertEqual(sina_fetch.call_count, 1)
        self.assertEqual(df["symbol"].tolist(), ["002074"])
        self.assertEqual(df["close"].tolist(), [10.4])

    def test_fetch_stock_daily_retries_with_alternate_proxy_mode(self) -> None:
        attempts: list[bool] = []
        raw_df = pd.DataFrame(
            {
                "日期": ["2026-04-18"],
                "开盘": [10.1],
                "最高": [10.5],
                "最低": [9.9],
                "收盘": [10.4],
                "成交量": [1000],
                "成交额": [1000000],
            }
        )

        def proxy_sensitive_fetch(**kwargs: object) -> pd.DataFrame:
            proxy_mode = bool(kwargs["use_system_proxy"])
            attempts.append(proxy_mode)
            if not proxy_mode:
                raise ConnectionError("direct connection failed")
            return raw_df

        with patch("tools.market_data._fetch_stock_daily_raw", side_effect=proxy_sensitive_fetch), patch(
            "tools.market_data._settings",
            return_value={"data": {"retry_count": 1, "retry_delay_seconds": 0, "use_system_proxy": False}},
        ):
            df = fetch_stock_daily("002074", "2026-04-01", "2026-04-18")

        self.assertEqual(attempts, [False, True])
        self.assertEqual(df["symbol"].tolist(), ["002074"])
        self.assertEqual(df["close"].tolist(), [10.4])


if __name__ == "__main__":
    unittest.main()
