from __future__ import annotations

import unittest

import pandas as pd

from tools.indicators import calc_amount_ratio, calc_basic_indicators, calc_pct_change


class IndicatorsSmokeTest(unittest.TestCase):
    def test_calc_basic_indicators(self) -> None:
        df = pd.DataFrame(
            {
                "close": [10, 11, 12, 13, 14, 15, 16],
                "high": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
                "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                "amount": [100, 120, 110, 130, 150, 170, 160],
            }
        )

        out = calc_basic_indicators(df)

        self.assertIn("ma5", out.columns)
        self.assertIn("atr14", out.columns)
        self.assertIn("amount_ratio_5d", out.columns)
        self.assertIn("pct_chg", out.columns)

    def test_calc_amount_ratio_avoids_divide_by_zero(self) -> None:
        df = pd.DataFrame({"amount": [0, 0, 0, 0, 0, 100]})

        out = calc_amount_ratio(df, 5)

        self.assertTrue(pd.isna(out.loc[5, "amount_ratio_5d"]))

    def test_calc_pct_change(self) -> None:
        df = pd.DataFrame({"close": [10, 11, 9.9]})

        out = calc_pct_change(df)

        self.assertTrue(pd.isna(out.loc[0, "pct_chg"]))
        self.assertAlmostEqual(out.loc[1, "pct_chg"], 0.1)
        self.assertAlmostEqual(out.loc[2, "pct_chg"], -0.1)


if __name__ == "__main__":
    unittest.main()
