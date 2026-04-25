from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from agents.data_agent import DataAgentOutput
from agents.research_agent import ResearchAgentOutput


class RiskDecision(BaseModel):
    symbol: str
    name: str
    category: Literal["position", "watchlist"]
    original_stance: str
    approved_stance: Literal["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]
    risk_level: Literal["low", "medium", "high"]
    position_pct_limit: float = Field(ge=0.0, le=1.0)
    stop_loss_rule: str
    take_profit_rule: str
    execution_guardrails: list[str] = Field(default_factory=list)
    downgrade_reason: str | None = None
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class RiskAgentOutput(BaseModel):
    review_date: str
    run_mode: Literal["deterministic_risk_agent"]
    portfolio_risk_summary: str
    symbols: list[RiskDecision]
    warnings: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _downgrade(current: str) -> str:
    order = ["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]
    if current not in order:
        return "普通观察"
    if current == "继续持有":
        return "谨慎持有"
    if current == "谨慎持有":
        return "减仓观察"
    if current == "重点观察":
        return "普通观察"
    if current == "普通观察":
        return "暂不关注"
    return current


def _build_stop_loss_rule(item: dict[str, Any], trade_rules: dict[str, Any]) -> str:
    stop_loss_pct = float(trade_rules.get("stop_loss_pct", 0.06))
    if item.get("is_position") and item.get("cost"):
        stop_price = float(item["cost"]) * (1 - stop_loss_pct)
        if item.get("ma20"):
            return f"固定止损参考 {stop_price:.2f}，同时关注 MA20={item['ma20']:.2f} 是否失守。"
        return f"固定止损参考 {stop_price:.2f}。"
    if item.get("ma10") and item.get("ma20"):
        return f"观察 MA10={item['ma10']:.2f} 与 MA20={item['ma20']:.2f} 的得失。"
    return f"按固定止损 {stop_loss_pct * 100:.1f}% 做风险预案。"


def _build_take_profit_rule(item: dict[str, Any], trade_rules: dict[str, Any]) -> str:
    take_profit_pct = float(trade_rules.get("take_profit_pct", 0.15))
    trailing_stop_pct = float(trade_rules.get("trailing_stop_pct", 0.08))
    if item.get("is_position"):
        return f"若浮盈继续扩大，可参考 {take_profit_pct * 100:.1f}% 分批兑现，并结合 {trailing_stop_pct * 100:.1f}% 回撤保护。"
    return "非持仓标的暂不设置止盈执行，仅保留观察计划。"


def _risk_from_item(item: dict[str, Any]) -> tuple[str, list[str]]:
    flags = list(item.get("risk_notes", []))
    if item.get("trend") == "weak":
        flags.append("趋势转弱，优先降低预期。")
        return "high", flags
    if item.get("is_position") and item.get("close", 0.0) < item.get("ma20", 0.0):
        flags.append("价格跌破 MA20，中期结构承压。")
        return "high", flags
    if item.get("is_position") and item.get("close", 0.0) < item.get("ma10", 0.0):
        flags.append("价格跌破 MA10，短线节奏转谨慎。")
        return "medium", flags
    if item.get("amount_ratio_5d", 1.0) <= 0.8:
        flags.append("量能偏弱，信号确认需要更多时间。")
        return "medium", flags
    return "low", flags


def run_risk_agent(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    config_dir: Path,
) -> RiskAgentOutput:
    rules = _read_yaml(config_dir / "risk_rules.yaml")
    portfolio_rules = rules.get("portfolio", {})
    trade_rules = rules.get("trade", {})
    min_cash_pct = float(trade_rules.get("min_cash_pct", 0.2))
    max_position_pct = float(portfolio_rules.get("max_position_pct", 0.2))
    reduced_position_pct = round(max_position_pct * 0.5, 4)

    research_map = {item.symbol: item for item in research_output.symbols}
    decisions: list[RiskDecision] = []

    for item in data_output.symbols:
        source = item.model_dump()
        research = research_map.get(item.symbol)
        original_stance = research.candidate_stance if research else (item.position_action or "普通观察")
        risk_level, risk_flags = _risk_from_item(source)
        approved_stance = original_stance
        downgrade_reason = None

        if risk_level == "high":
            next_stance = _downgrade(original_stance)
            approved_stance = _downgrade(next_stance) if item.is_position and original_stance == "继续持有" else next_stance
            position_pct_limit = 0.0 if not item.is_position else reduced_position_pct
            downgrade_reason = "风险约束触发高风险处理，需优先控制回撤。"
        elif risk_level == "medium":
            approved_stance = _downgrade(original_stance)
            position_pct_limit = reduced_position_pct if approved_stance in {"继续持有", "谨慎持有", "重点观察"} else 0.0
            downgrade_reason = "风险约束提示节奏放缓，先降低动作级别。"
        else:
            position_pct_limit = max_position_pct if approved_stance in {"继续持有", "重点观察"} else reduced_position_pct
            if not item.is_position and approved_stance not in {"重点观察", "普通观察"}:
                approved_stance = "重点观察" if item.trend == "up" else "普通观察"

        guardrails = [
            f"单标的上限不超过 {max_position_pct * 100:.0f}%",
            f"组合至少保留 {min_cash_pct * 100:.0f}% 现金缓冲",
            "仅输出计划，不直接触发交易执行",
        ]
        decisions.append(
            RiskDecision(
                symbol=item.symbol,
                name=item.name,
                category="position" if item.is_position else "watchlist",
                original_stance=original_stance,
                approved_stance=approved_stance,
                risk_level=risk_level,
                position_pct_limit=position_pct_limit,
                stop_loss_rule=_build_stop_loss_rule(source, trade_rules),
                take_profit_rule=_build_take_profit_rule(source, trade_rules),
                execution_guardrails=guardrails,
                downgrade_reason=downgrade_reason,
                risk_flags=risk_flags,
                confidence=0.72 if risk_level == "low" else 0.82,
            )
        )

    high_risk = sum(1 for item in decisions if item.risk_level == "high")
    medium_risk = sum(1 for item in decisions if item.risk_level == "medium")
    if high_risk:
        portfolio_risk_summary = "组合中存在高风险标的，次日应优先处理防守与减仓观察。"
    elif medium_risk:
        portfolio_risk_summary = "组合整体可跟踪，但已有标的进入谨慎区，仓位节奏宜放缓。"
    else:
        portfolio_risk_summary = "组合风险总体可控，重点看强势标的是否延续并守住关键均线。"

    return RiskAgentOutput(
        review_date=data_output.review_date,
        run_mode="deterministic_risk_agent",
        portfolio_risk_summary=portfolio_risk_summary,
        symbols=decisions,
        warnings=list(data_output.warnings) + list(research_output.warnings),
    )
