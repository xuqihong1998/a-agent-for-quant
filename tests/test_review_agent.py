from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.review_agent import DailyResearchReview, _coerce_review_payload, build_rule_fallback_review, run_review_agent
from tools.agent_review import merge_agent_output, render_agent_daily_report, write_agent_review_json


SAMPLE_RESULTS = [
    {
        "symbol": "002463",
        "name": "沪电股份",
        "is_position": True,
        "position_name": "沪电股份",
        "position_action": "继续持有",
        "close": 106.62,
        "pct_chg": 0.021,
        "ma5": 101.07,
        "ma10": 96.71,
        "ma20": 88.01,
        "atr14": 4.21,
        "amount_ratio_5d": 1.37,
        "volume_status": "放量",
        "trend": "up",
        "comment": "趋势保持，关注强势延续",
        "next_focus": "关注 MA5=101.07 与 MA10=96.71 附近承接",
        "action_hint": "持有，回踩关键均线不破可继续跟踪",
        "risk_notes": ["暂无明显风险信号"],
        "cost": 98.83,
        "shares": 2000,
        "market_value": 213240.0,
        "pnl_pct": 0.0788,
        "pnl_amount": 15582.0,
    },
    {
        "symbol": "002074",
        "name": "国轩高科",
        "is_position": False,
        "close": 39.76,
        "pct_chg": -0.004,
        "ma5": 40.20,
        "ma10": 39.77,
        "ma20": 37.67,
        "atr14": 1.35,
        "amount_ratio_5d": 0.73,
        "volume_status": "缩量",
        "trend": "neutral",
        "comment": "量能偏弱，耐心等待突破",
        "next_focus": "观察能否站稳 MA10=39.77",
        "action_hint": "先观察，不急于出手",
    },
]


class ReviewAgentTest(unittest.TestCase):
    def test_build_rule_fallback_review(self) -> None:
        review = build_rule_fallback_review(SAMPLE_RESULTS, "2026-04-24", warning="sdk missing")

        self.assertEqual(review.run_mode, "rule_fallback")
        self.assertEqual(review.review_date, "2026-04-24")
        self.assertEqual(len(review.symbols), 2)
        self.assertEqual(review.symbols[0].symbol, "002463")
        self.assertEqual(review.warnings, ["sdk missing"])

    def test_merge_agent_output(self) -> None:
        review = DailyResearchReview(
            review_date="2026-04-24",
            run_mode="openai_agent",
            market_summary="强势股分化但趋势未坏。",
            portfolio_summary="持仓仍以趋势跟踪为主。",
            next_day_overview="先看核心标的短均线承接。",
            symbols=[
                {
                    "symbol": "002463",
                    "name": "沪电股份",
                    "category": "position",
                    "stance": "继续持有",
                    "headline": "趋势仍强，优先看短均线承接",
                    "thesis": "股价维持在多条均线上方，放量后仍有延续预期。",
                    "key_signals": ["站稳 MA5/MA10", "量能维持放大"],
                    "risk_flags": ["若回落并跌破 MA10，强势节奏会放缓。"],
                    "next_day_plan": "观察 MA5 附近回踩承接，不追高。",
                    "focus_price": "101.07 ~ 106.62",
                    "invalidation": "跌破 MA10 且放量走弱。",
                    "confidence": 0.81,
                }
            ],
        )

        merged = merge_agent_output(SAMPLE_RESULTS, review.model_dump())

        self.assertEqual(merged[0]["agent_headline"], "趋势仍强，优先看短均线承接")
        self.assertEqual(merged[0]["agent_action"], "继续持有")
        self.assertEqual(merged[1]["agent_headline"], SAMPLE_RESULTS[1]["comment"])

    def test_render_and_write_agent_outputs(self) -> None:
        review = build_rule_fallback_review(SAMPLE_RESULTS, "2026-04-24")

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "2026-04-24.md"
            json_path = Path(tmp_dir) / "2026-04-24.json"
            render_agent_daily_report(SAMPLE_RESULTS, review.model_dump(), report_path)
            write_agent_review_json(SAMPLE_RESULTS, review.model_dump(), json_path)

            report_text = report_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertIn("# 盘后复盘报告（2026-04-24）", report_text)
        self.assertIn("- 运行模式：rule_fallback", report_text)
        self.assertIn("- AI 研究结论：", report_text)
        self.assertIn("\"run_mode\": \"rule_fallback\"", json_text)
        self.assertIn("\"results\"", json_text)

    def test_run_review_agent_falls_back_when_sdk_fails(self) -> None:
        with patch("agents.review_agent.build_review_context", return_value=({"summary_items": []}, SAMPLE_RESULTS)), patch(
            "agents.review_agent._import_openai_agents_sdk",
            side_effect=ImportError("No module named agents"),
        ):
            review, base_results = run_review_agent(
                config_dir=Path("."),
                data_dir=Path("."),
                review_date="2026-04-24",
                allow_rule_fallback=True,
            )

        self.assertEqual(review.run_mode, "rule_fallback")
        self.assertEqual(base_results, SAMPLE_RESULTS)

    def test_run_review_agent_retries_without_structured_output_when_json_mode_is_unsupported(self) -> None:
        class FakeResult:
            def __init__(self, final_output: object) -> None:
                self.final_output = final_output

        class FakeRunner:
            calls = 0

            @staticmethod
            def run_sync(agent, prompt, run_config=None):
                FakeRunner.calls += 1
                if FakeRunner.calls == 1:
                    raise Exception("Json mode is not supported for this model.")
                return FakeResult(
                    """{
  "review_date": "2026-04-24",
  "run_mode": "openai_agent",
  "market_summary": "test",
  "portfolio_summary": "test",
  "next_day_overview": "test",
  "symbols": [
    {
      "symbol": "002463",
      "name": "沪电股份",
      "category": "position",
      "stance": "持有",
      "headline": "test",
      "thesis": "test",
      "key_signals": ["a"],
      "risk_flags": ["b"],
      "next_day_plan": "test",
      "focus_price": "test",
      "invalidation": "test",
      "confidence": 0.8
    }
  ],
  "warnings": []
}"""
                )

        class FakeSDK:
            Runner = FakeRunner
            RunConfig = None
            MultiProvider = None

            @staticmethod
            def Agent(**kwargs):
                return kwargs

            @staticmethod
            def function_tool(fn):
                return fn

        with patch("agents.review_agent.build_review_context", return_value=({"summary_items": []}, SAMPLE_RESULTS)), patch(
            "agents.review_agent._import_openai_agents_sdk",
            return_value=FakeSDK,
        ):
            review, base_results = run_review_agent(
                config_dir=Path("."),
                data_dir=Path("."),
                review_date="2026-04-24",
                allow_rule_fallback=True,
            )

        self.assertEqual(review.run_mode, "openai_agent")
        self.assertEqual(review.symbols[0].stance, "继续持有")
        self.assertEqual(base_results, SAMPLE_RESULTS)

    def test_coerce_review_payload_normalizes_common_model_shape_errors(self) -> None:
        payload = {
            "review_date": "2026-04-24",
            "run_mode": "openai_agent",
            "market_summary": {"趋势概览": "偏强"},
            "portfolio_summary": {"持仓数量": 3},
            "next_day_overview": "test",
            "warnings": [],
            "symbols": [
                {
                    "symbol": "002463",
                    "name": "沪电股份",
                    "category": "PCB/AI服务器",
                    "stance": "持有",
                    "headline": "test",
                    "thesis": "test",
                    "key_signals": ["a"],
                    "risk_flags": ["b"],
                    "next_day_plan": "test",
                    "focus_price": 102.02,
                    "invalidation": "test",
                    "confidence": "0.8",
                }
            ],
        }

        normalized = _coerce_review_payload(payload, SAMPLE_RESULTS)

        self.assertEqual(normalized["market_summary"], "趋势概览: 偏强")
        self.assertEqual(normalized["portfolio_summary"], "持仓数量: 3")
        self.assertEqual(normalized["symbols"][0]["category"], "position")
        self.assertEqual(normalized["symbols"][0]["stance"], "继续持有")
        self.assertEqual(normalized["symbols"][0]["focus_price"], "102.02")


if __name__ == "__main__":
    unittest.main()
