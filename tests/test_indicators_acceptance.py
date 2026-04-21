from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from tools.indicators import calc_amount_change, calc_atr, calc_ma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDICATOR_COLUMNS = ["ma5", "ma10", "ma20", "atr14", "amount_ratio_5d"]


class IndicatorAcceptanceTest(unittest.TestCase):
    def test_can_output_latest_indicator_values_for_one_stock(self) -> None:
        parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            self.skipTest(f"no parquet files found in {RAW_DATA_DIR}")

        df = pd.read_parquet(parquet_files[0])
        if len(df) < 21:
            self.skipTest("need at least 21 rows to calculate all acceptance indicators")

        df = calc_ma(df, 5)
        df = calc_ma(df, 10)
        df = calc_ma(df, 20)
        df = calc_atr(df, 14)
        df = calc_amount_change(df, 5)

        latest = df[["date", "symbol", *INDICATOR_COLUMNS]].tail(1)
        print(latest.to_string(index=False))

        self.assertTrue(set(INDICATOR_COLUMNS).issubset(df.columns))
        self.assertTrue(latest[INDICATOR_COLUMNS].notna().all(axis=None))


if __name__ == "__main__":
    unittest.main()
