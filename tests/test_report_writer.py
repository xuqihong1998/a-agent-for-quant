from __future__ import annotations

import unittest

import pandas as pd

from tools.report_writer import summarize_symbol


class ReportWriterSmokeTest(unittest.TestCase):
    def test_summarize_symbol(self) -> None:
        df = pd.DataFrame(
            {
                "close": [10, 11, 12],
                "ma5": [9, 10, 11],
                "ma10": [8, 9, 10],
                "ma20": [7, 8, 9],
                "atr14": [1, 1, 1],
                "amount_ratio_5d": [1.0, 1.1, 1.3],
            }
        )

        item = summarize_symbol("000001", df)

        self.assertEqual(item["symbol"], "000001")
        self.assertIn(item["trend"], ("up", "neutral", "weak"))


if __name__ == "__main__":
    unittest.main()
