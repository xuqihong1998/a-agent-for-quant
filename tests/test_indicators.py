from __future__ import annotations

import unittest

import pandas as pd

from tools.indicators import calc_basic_indicators


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


if __name__ == "__main__":
    unittest.main()
