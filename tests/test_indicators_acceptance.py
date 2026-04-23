from __future__ import annotations

from pathlib import Path
import unittest

from tools.indicators import calc_basic_indicators
from tools.market_data import load_symbol_df


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDICATOR_COLUMNS = ["ma5", "ma10", "ma20", "atr14", "amount_ratio_5d", "pct_chg"]


class IndicatorAcceptanceTest(unittest.TestCase):
    def test_can_output_latest_indicator_values_for_one_stock(self) -> None:
        parquet_files = sorted(RAW_DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            self.skipTest(f"no parquet files found in {RAW_DATA_DIR}")

        df = load_symbol_df(parquet_files[0].stem)
        if len(df) < 21:
            self.skipTest("need at least 21 rows to calculate all acceptance indicators")

        df = calc_basic_indicators(df)

        latest = df[["date", "symbol", *INDICATOR_COLUMNS]].tail(1)
        print(latest.to_string(index=False))

        self.assertTrue(set(INDICATOR_COLUMNS).issubset(df.columns))
        self.assertTrue(latest[INDICATOR_COLUMNS].notna().all(axis=None))


if __name__ == "__main__":
    unittest.main()
