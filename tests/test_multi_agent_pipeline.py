from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.data_agent import DataAgentOutput, DataSymbolSummary
from agents.graph import (
    _apply_human_review_decision,
    _build_human_review_payload,
    _load_context_node,
    _route_after_plan,
    run_sequential_agent_pipeline,
)
from agents.plan_agent import PlanAgentOutput, _normalize_plan_payload, build_rule_fallback_plan
from agents.research_agent import ResearchAgentOutput, ResearchOpinion, build_rule_fallback_research
from agents.risk_agent import RiskAgentOutput, RiskDecision, run_risk_agent
from agents.sdk_support import run_structured_agent
from tools.agent_pipeline import merge_pipeline_results, render_pipeline_daily_report, render_pipeline_plan, write_pipeline_json


def make_data_output() -> DataAgentOutput:
    return DataAgentOutput(
        review_date="2026-04-25",
        run_mode="deterministic_data_agent",
        summary_stats={"position_count": 1, "watch_count": 1, "up_count": 1, "neutral_count": 1, "weak_count": 0},
        warnings=[],
        symbols=[
            DataSymbolSummary(
                symbol="002463",
                name="沪电股份",
                category="position",
                tags=["PCB", "AI服务器"],
                date="2026-04-25",
                is_position=True,
                close=106.62,
                pct_chg=0.021,
                ma5=101.07,
                ma10=96.71,
                ma20=88.01,
                atr14=4.21,
                amount_ratio_5d=1.37,
                volume_status="放量",
                trend="up",
                comment="趋势保持，关注强势延续",
                next_focus="关注 MA5=101.07 与 MA10=96.71 附近承接",
                action_hint="持有，回踩关键均线不破可继续跟踪",
                position_action="继续持有",
                cost=98.83,
                shares=2000,
                market_value=213240.0,
                pnl_pct=0.0788,
                pnl_amount=15582.0,
                risk_notes=["暂无明显风险信号"],
            ),
            DataSymbolSummary(
                symbol="002074",
                name="国轩高科",
                category="watchlist",
                tags=["新能源", "动力电池"],
                date="2026-04-25",
                is_position=False,
                close=39.76,
                pct_chg=-0.004,
                ma5=40.20,
                ma10=39.77,
                ma20=37.67,
                atr14=1.35,
                amount_ratio_5d=0.73,
                volume_status="缩量",
                trend="neutral",
                comment="量能偏弱，耐心等待突破",
                next_focus="观察能否站稳 MA10=39.77",
                action_hint="先观察，不急于出手",
            ),
        ],
    )


class MultiAgentPipelineTest(unittest.TestCase):
    def test_rule_fallback_research_and_risk(self) -> None:
        data_output = make_data_output()
        research_output = build_rule_fallback_research(data_output)
        risk_output = run_risk_agent(data_output, research_output, Path("config"))

        self.assertEqual(research_output.run_mode, "rule_fallback")
        self.assertEqual(risk_output.run_mode, "deterministic_risk_agent")
        self.assertEqual(len(risk_output.symbols), 2)

    def test_rule_fallback_plan(self) -> None:
        data_output = make_data_output()
        research_output = build_rule_fallback_research(data_output)
        risk_output = RiskAgentOutput(
            review_date="2026-04-25",
            run_mode="deterministic_risk_agent",
            portfolio_risk_summary="组合风险总体可控。",
            warnings=[],
            symbols=[
                RiskDecision(
                    symbol="002463",
                    name="沪电股份",
                    category="position",
                    original_stance="继续持有",
                    approved_stance="继续持有",
                    risk_level="low",
                    position_pct_limit=0.2,
                    stop_loss_rule="test",
                    take_profit_rule="test",
                    execution_guardrails=["test"],
                    risk_flags=["暂无明显风险信号"],
                    confidence=0.8,
                )
            ],
        )

        plan_output = build_rule_fallback_plan(data_output, research_output, risk_output)

        self.assertEqual(plan_output.run_mode, "rule_fallback")
        self.assertEqual(plan_output.symbols[0].final_action, "继续持有")

    def test_plan_payload_is_clamped_by_risk_upper_bound(self) -> None:
        data_output = make_data_output()
        research_output = build_rule_fallback_research(data_output)
        risk_output = RiskAgentOutput(
            review_date="2026-04-25",
            run_mode="deterministic_risk_agent",
            portfolio_risk_summary="test",
            warnings=[],
            symbols=[
                RiskDecision(
                    symbol="002463",
                    name="沪电股份",
                    category="position",
                    original_stance="继续持有",
                    approved_stance="谨慎持有",
                    risk_level="medium",
                    position_pct_limit=0.1,
                    stop_loss_rule="test",
                    take_profit_rule="test",
                    execution_guardrails=["test"],
                    risk_flags=["b"],
                    confidence=0.8,
                )
            ],
        )

        normalized = _normalize_plan_payload(
            {
                "review_date": "2026-04-25",
                "run_mode": "openai_agent",
                "market_summary": "test",
                "portfolio_summary": "test",
                "next_day_overview": "test",
                "warnings": [],
                "symbols": [
                    {
                        "symbol": "002463",
                        "name": "沪电股份",
                        "category": "position",
                        "final_action": "继续持有",
                        "headline": "test",
                        "thesis": "test",
                        "plan_steps": ["a"],
                        "focus_price": "test",
                        "invalidation": "test",
                        "risk_level": "medium",
                        "position_pct_limit": 0.2,
                        "risk_flags": ["b"],
                        "confidence": 0.9,
                    }
                ],
            },
            data_output,
            research_output,
            risk_output,
        )

        self.assertEqual(normalized["symbols"][0]["final_action"], "谨慎持有")

    def test_graph_human_review_payload_and_apply_decision(self) -> None:
        data_output = make_data_output()
        research_output = build_rule_fallback_research(data_output)
        risk_output = run_risk_agent(data_output, research_output, Path("config"))
        plan_output = build_rule_fallback_plan(data_output, research_output, risk_output)

        payload = _build_human_review_payload(
            {
                "plan_output": plan_output.model_dump(),
                "risk_output": risk_output.model_dump(),
            }
        )
        self.assertEqual(payload["step"], "human_review")
        self.assertEqual(payload["review_date"], "2026-04-25")
        self.assertTrue(payload["actions"])

        updated_plan, decision, final_status = _apply_human_review_decision(
            plan_output,
            {
                "approved": True,
                "notes": "人工确认通过",
                "edited_next_day_overview": "开盘前优先检查高风险持仓，再决定是否执行原计划。",
            },
        )
        self.assertEqual(final_status, "approved")
        self.assertEqual(decision["approved"], True)
        self.assertIn("开盘前优先检查高风险持仓", updated_plan.next_day_overview)

    def test_graph_load_context_node(self) -> None:
        state = _load_context_node(
            {
                "review_date": "2026-04-25",
                "config_dir": "config",
                "data_dir": "data/raw",
            }
        )
        self.assertEqual(state["context_loaded"], True)
        self.assertEqual(state["review_date"], "2026-04-25")

    def test_route_after_plan_can_skip_human_review(self) -> None:
        self.assertEqual(_route_after_plan({"human_review_enabled": False}), "END")
        self.assertEqual(_route_after_plan({"human_review_enabled": True}), "human_review")

    def test_run_structured_agent_can_start_in_text_json_mode(self) -> None:
        call_flags: list[bool] = []

        class FakeResult:
            def __init__(self, final_output: object) -> None:
                self.final_output = final_output

        class FakeRunner:
            @staticmethod
            def run_sync(agent, prompt, run_config=None):
                return FakeResult('{"value": "ok"}')

        class FakeSDK:
            Runner = FakeRunner

        with patch("agents.sdk_support.import_openai_agents_sdk", return_value=FakeSDK):
            result = run_structured_agent(
                create_agent=lambda structured_output: call_flags.append(structured_output) or {"structured_output": structured_output},
                prompt="test",
                output_model=type("SimpleModel", (), {"model_validate": staticmethod(lambda payload: payload)}),
                prefer_text_json=True,
            )

        self.assertEqual(call_flags, [False])
        self.assertEqual(result["value"], "ok")

    def test_merge_and_render_pipeline_outputs(self) -> None:
        data_output = make_data_output()
        research_output = ResearchAgentOutput(
            review_date="2026-04-25",
            run_mode="rule_fallback",
            market_summary="test",
            portfolio_summary="test",
            warnings=[],
            symbols=[
                ResearchOpinion(
                    symbol="002463",
                    name="沪电股份",
                    category="position",
                    theme_view="PCB/AI服务器",
                    pattern_view="多头结构保持",
                    event_view="放量上行",
                    candidate_stance="继续持有",
                    headline="趋势仍强",
                    thesis="test",
                    key_signals=["a"],
                    risk_flags=["b"],
                    focus_price="101-106",
                    next_day_plan="test",
                    invalidation="test",
                    confidence=0.8,
                )
            ],
        )
        risk_output = RiskAgentOutput(
            review_date="2026-04-25",
            run_mode="deterministic_risk_agent",
            portfolio_risk_summary="test",
            warnings=[],
            symbols=[
                RiskDecision(
                    symbol="002463",
                    name="沪电股份",
                    category="position",
                    original_stance="继续持有",
                    approved_stance="继续持有",
                    risk_level="low",
                    position_pct_limit=0.2,
                    stop_loss_rule="test",
                    take_profit_rule="test",
                    execution_guardrails=["test"],
                    risk_flags=["b"],
                    confidence=0.8,
                )
            ],
        )
        plan_output = PlanAgentOutput(
            review_date="2026-04-25",
            run_mode="rule_fallback",
            market_summary="test",
            portfolio_summary="test",
            next_day_overview="test",
            warnings=[],
            symbols=[
                {
                    "symbol": "002463",
                    "name": "沪电股份",
                    "category": "position",
                    "final_action": "继续持有",
                    "headline": "趋势仍强",
                    "thesis": "test",
                    "plan_steps": ["step1", "step2"],
                    "focus_price": "101-106",
                    "invalidation": "test",
                    "risk_level": "low",
                    "position_pct_limit": 0.2,
                    "risk_flags": ["b"],
                    "confidence": 0.8,
                }
            ],
        )

        merged = merge_pipeline_results(data_output, research_output, risk_output, plan_output)
        self.assertEqual(len(merged), 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "2026-04-25.md"
            plan_path = Path(tmp_dir) / "2026-04-25-plan.md"
            json_path = Path(tmp_dir) / "2026-04-25.pipeline.json"
            render_pipeline_daily_report(data_output, research_output, risk_output, plan_output, report_path)
            render_pipeline_plan(plan_output, plan_path)
            write_pipeline_json(data_output, research_output, risk_output, plan_output, json_path)

            report_text = report_path.read_text(encoding="utf-8")
            plan_text = plan_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertIn("## AI 总结", report_text)
        self.assertIn("## 继续持有", plan_text)
        self.assertIn("\"data_agent\"", json_text)
        self.assertIn("\"plan_agent\"", json_text)

    def test_sequential_pipeline_orchestrates_all_stages(self) -> None:
        data_output = make_data_output()
        research_output = build_rule_fallback_research(data_output)
        risk_output = run_risk_agent(data_output, research_output, Path("config"))
        plan_output = build_rule_fallback_plan(data_output, research_output, risk_output)

        with patch("agents.graph.run_data_agent", return_value=data_output), patch(
            "agents.graph.run_research_agent",
            return_value=research_output,
        ), patch("agents.graph.run_risk_agent", return_value=risk_output), patch(
            "agents.graph.run_plan_agent",
            return_value=plan_output,
        ):
            outputs = run_sequential_agent_pipeline(
                config_dir=Path("config"),
                data_dir=Path("data/raw"),
                review_date="2026-04-25",
                prefer_text_json=True,
                human_review_enabled=False,
            )

        self.assertEqual(outputs[0].run_mode, "deterministic_data_agent")
        self.assertEqual(outputs[1].review_date, "2026-04-25")
        self.assertEqual(outputs[2].run_mode, "deterministic_risk_agent")
        self.assertEqual(outputs[3].review_date, "2026-04-25")


if __name__ == "__main__":
    unittest.main()
