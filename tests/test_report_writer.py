from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from tools.report_writer import (
    build_symbol_comment,
    enrich_position_info,
    format_position_markdown,
    format_symbol_markdown,
    judge_trend,
    judge_volume_status,
    render_daily_report,
    summarize_symbol,
)


class ReportWriterSmokeTest(unittest.TestCase):
    def test_judge_trend(self) -> None:
        up_row = pd.Series({"close": 12, "ma5": 11, "ma10": 10, "ma20": 9})
        weak_row = pd.Series({"close": 8, "ma5": 8.5, "ma10": 9, "ma20": 10})
        neutral_row = pd.Series({"close": 10, "ma5": 9, "ma10": 10.5, "ma20": 9.5})

        self.assertEqual(judge_trend(up_row), "up")
        self.assertEqual(judge_trend(weak_row), "weak")
        self.assertEqual(judge_trend(neutral_row), "neutral")

    def test_judge_volume_status(self) -> None:
        self.assertEqual(judge_volume_status(pd.Series({"amount_ratio_5d": 1.2})), "放量")
        self.assertEqual(judge_volume_status(pd.Series({"amount_ratio_5d": 0.8})), "缩量")
        self.assertEqual(judge_volume_status(pd.Series({"amount_ratio_5d": 1.0})), "平量")

    def test_build_symbol_comment(self) -> None:
        summary = {"trend": "up", "volume_status": "放量"}
        self.assertEqual(build_symbol_comment(summary), "趋势保持，关注强势延续")

    def test_summarize_symbol_adds_next_day_fields(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2026-04-18", "2026-04-21"],
                "close": [10, 11],
                "high": [10.5, 11.3],
                "pct_chg": [0.01, 0.02],
                "ma5": [9, 10],
                "ma10": [8, 9],
                "ma20": [7, 8],
                "atr14": [1, 1.2],
                "amount_ratio_5d": [1.0, 1.3],
            }
        )

        item = summarize_symbol("1", df)

        self.assertEqual(item["symbol"], "000001")
        self.assertEqual(item["trend"], "up")
        self.assertEqual(item["volume_status"], "放量")
        self.assertEqual(item["comment"], "趋势保持，关注强势延续")
        self.assertIn("MA5=10.00", item["next_focus"])
        self.assertEqual(item["support_level"], 9.0)
        self.assertEqual(item["pressure_level"], 11.3)
        self.assertEqual(item["action_hint"], "回踩不破可继续关注")

    def test_enrich_position_info_overrides_action_hint(self) -> None:
        item = {
            "symbol": "002074",
            "name": "国轩高科",
            "close": 38.0,
            "high": 38.4,
            "pct_chg": -0.025,
            "ma5": 39.0,
            "ma10": 38.5,
            "ma20": 38.2,
            "atr14": 1.35,
            "amount_ratio_5d": 1.3,
            "volume_status": "放量",
            "trend": "weak",
            "comment": "警惕放量走弱",
            "next_focus": "留意 MA20=38.20 是否失守",
            "support_level": 38.2,
            "pressure_level": 38.5,
            "action_hint": "跌破关键位则降级观察",
        }
        position = {"symbol": "002074", "name": "国轩高科", "cost": "40.0", "shares": "1000"}

        enriched = enrich_position_info(item, position)

        self.assertTrue(enriched["is_position"])
        self.assertEqual(enriched["cost"], 40.0)
        self.assertEqual(enriched["shares"], 1000)
        self.assertEqual(enriched["market_value"], 38000.0)
        self.assertAlmostEqual(enriched["pnl_pct"], -0.05)
        self.assertEqual(enriched["pnl_amount"], -2000.0)
        self.assertEqual(enriched["position_action"], "减仓观察")
        self.assertEqual(enriched["action_hint"], "减仓观察，跌破关键位则进一步降级处理")
        self.assertIn("跌破MA10，短线转弱", enriched["risk_notes"])
        self.assertIn("跌破MA20，中期趋势受压", enriched["risk_notes"])
        self.assertIn("放量下跌，警惕资金流出", enriched["risk_notes"])

    def test_format_markdown_contains_next_day_lines(self) -> None:
        symbol_item = {
            "symbol": "002463",
            "name": "沪电股份",
            "close": 98.00,
            "pct_chg": 0.0035,
            "ma5": 96.08,
            "ma10": 90.37,
            "ma20": 84.77,
            "atr14": 4.68,
            "amount_ratio_5d": 0.86,
            "volume_status": "平量",
            "trend": "up",
            "comment": "趋势保持，继续观察均线支撑",
            "next_focus": "关注 MA5=96.08 与 MA10=90.37 附近承接",
            "action_hint": "回踩不破可继续关注",
        }
        position_item = {
            **symbol_item,
            "symbol": "002074",
            "name": "国轩高科",
            "position_name": "国轩高科",
            "is_position": True,
            "cost": 39.40,
            "shares": 1000,
            "market_value": 39940.0,
            "pnl_pct": 0.0136,
            "pnl_amount": 537.0,
            "risk_notes": ["暂无明显风险信号"],
            "position_action": "持有",
            "action_hint": "持有，回踩关键均线不破可继续跟踪",
        }

        symbol_markdown = format_symbol_markdown(symbol_item)
        position_markdown = format_position_markdown(position_item)

        self.assertIn("- 明日观察点：关注 MA5=96.08 与 MA10=90.37 附近承接", symbol_markdown)
        self.assertIn("- 动作建议：回踩不破可继续关注", symbol_markdown)
        self.assertIn("- 持仓动作：持有", position_markdown)
        self.assertIn("- 动作建议：持有，回踩关键均线不破可继续跟踪", position_markdown)

    def test_render_daily_report(self) -> None:
        results = [
            {
                "symbol": "002074",
                "name": "国轩高科",
                "position_name": "国轩高科",
                "is_position": True,
                "close": 39.94,
                "pct_chg": -0.0072,
                "ma5": 39.72,
                "ma10": 38.11,
                "ma20": 37.20,
                "atr14": 1.35,
                "amount_ratio_5d": 0.64,
                "volume_status": "缩量",
                "trend": "up",
                "comment": "趋势未坏，但量能一般",
                "next_focus": "关注 MA5=39.72 与 MA10=38.11 附近承接",
                "action_hint": "持有，回踩关键均线不破可继续跟踪",
                "cost": 39.40,
                "shares": 1000,
                "market_value": 39940.0,
                "pnl_pct": 0.0136,
                "pnl_amount": 537.0,
                "risk_notes": ["暂无明显风险信号"],
                "position_action": "持有",
            },
            {
                "symbol": "002463",
                "name": "沪电股份",
                "is_position": False,
                "close": 98.00,
                "pct_chg": 0.0035,
                "ma5": 96.08,
                "ma10": 90.37,
                "ma20": 84.77,
                "atr14": 4.68,
                "amount_ratio_5d": 0.86,
                "volume_status": "平量",
                "trend": "up",
                "comment": "趋势保持，继续观察均线支撑",
                "next_focus": "关注 MA5=96.08 与 MA10=90.37 附近承接",
                "action_hint": "回踩不破可继续关注",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "2026-04-21.md"
            render_daily_report(results, output_path)
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("# 盘后复盘报告（2026-04-21）", content)
        self.assertIn("## 持仓分析", content)
        self.assertIn("## 观察池分析", content)
        self.assertIn("- 浮盈亏 %：1.36%", content)
        self.assertIn("- 明日观察点：关注 MA5=39.72 与 MA10=38.11 附近承接", content)
        self.assertIn("- 动作建议：回踩不破可继续关注", content)


if __name__ == "__main__":
    unittest.main()
