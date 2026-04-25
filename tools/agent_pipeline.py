from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.data_agent import DataAgentOutput
from agents.plan_agent import PlanAgentOutput
from agents.research_agent import ResearchAgentOutput
from agents.risk_agent import RiskAgentOutput


PLAN_ACTIONS = ["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]


def _log(message: str) -> None:
    print(f"[PIPELINE] {message}")


def merge_pipeline_results(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
    plan_output: PlanAgentOutput,
) -> list[dict[str, Any]]:
    _log(
        "merge_pipeline_results "
        f"data={len(data_output.symbols)} research={len(research_output.symbols)} "
        f"risk={len(risk_output.symbols)} plan={len(plan_output.symbols)}"
    )
    research_map = {item.symbol: item for item in research_output.symbols}
    risk_map = {item.symbol: item for item in risk_output.symbols}
    plan_map = {item.symbol: item for item in plan_output.symbols}

    merged: list[dict[str, Any]] = []
    for item in data_output.symbols:
        research_item = research_map.get(item.symbol)
        risk_item = risk_map.get(item.symbol)
        plan_item = plan_map.get(item.symbol)
        merged.append(
            {
                **item.model_dump(),
                "research": research_item.model_dump() if research_item else None,
                "risk": risk_item.model_dump() if risk_item else None,
                "plan": plan_item.model_dump() if plan_item else None,
            }
        )
    return merged


def render_pipeline_daily_report(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
    plan_output: PlanAgentOutput,
    output_path: Path,
    title: str = "盘后复盘报告",
) -> None:
    _log(f"render_pipeline_daily_report start path={output_path}")
    merged = merge_pipeline_results(data_output, research_output, risk_output, plan_output)
    positions = [item for item in merged if item.get("is_position")]
    watch_only = [item for item in merged if not item.get("is_position")]

    body = [
        f"# {title}（{plan_output.review_date}）",
        "",
        "## 总览",
        f"- data_agent：{data_output.run_mode}",
        f"- research_agent：{research_output.run_mode}",
        f"- risk_agent：{risk_output.run_mode}",
        f"- plan_agent：{plan_output.run_mode}",
        f"- 趋势向上：{data_output.summary_stats.get('up_count', 0)}",
        f"- 趋势中性：{data_output.summary_stats.get('neutral_count', 0)}",
        f"- 趋势转弱：{data_output.summary_stats.get('weak_count', 0)}",
        "",
        "## AI 总结",
        f"- 市场概览：{plan_output.market_summary}",
        f"- 持仓视角：{plan_output.portfolio_summary}",
        f"- 次日总计划：{plan_output.next_day_overview}",
        "",
        "## 持仓分析",
    ]

    if not positions:
        body.extend(["- 当前无持仓分析对象", ""])
    else:
        for item in positions:
            research = item.get("research") or {}
            risk = item.get("risk") or {}
            plan = item.get("plan") or {}
            risk_text = "；".join(risk.get("risk_flags", [])) if risk.get("risk_flags") else "暂无明显风险提示"
            body.extend(
                [
                    f"### {item['symbol']} {item['name']}",
                    f"- 收盘价：{item['close']:.2f}",
                    f"- 趋势判断：{item['trend']}",
                    f"- 研究主题：{research.get('theme_view', '暂无')}",
                    f"- 研究形态：{research.get('pattern_view', '暂无')}",
                    f"- 研究触发：{research.get('event_view', '暂无')}",
                    f"- 候选观点：{research.get('candidate_stance', '暂无')}",
                    f"- 风控结论：{risk.get('approved_stance', '暂无')} ({risk.get('risk_level', 'unknown')})",
                    f"- 风险提示：{risk_text}",
                    f"- 次日计划：{'；'.join(plan.get('plan_steps', [])) if plan.get('plan_steps') else '暂无'}",
                    f"- 关注价位：{plan.get('focus_price', item['next_focus'])}",
                    f"- 失效条件：{plan.get('invalidation', '暂无')}",
                    "",
                ]
            )

    body.append("## 观察池分析")
    if not watch_only:
        body.extend(["- 当前无观察池标的", ""])
    else:
        for item in watch_only:
            research = item.get("research") or {}
            risk = item.get("risk") or {}
            plan = item.get("plan") or {}
            body.extend(
                [
                    f"### {item['symbol']} {item['name']}",
                    f"- 收盘价：{item['close']:.2f}",
                    f"- 趋势判断：{item['trend']}",
                    f"- 主题标签：{'/'.join(item.get('tags', [])) if item.get('tags') else '暂无'}",
                    f"- 研究标题：{research.get('headline', item['comment'])}",
                    f"- 研究结论：{research.get('thesis', item['comment'])}",
                    f"- 风控结论：{risk.get('approved_stance', '暂无')} ({risk.get('risk_level', 'unknown')})",
                    f"- 次日计划：{'；'.join(plan.get('plan_steps', [])) if plan.get('plan_steps') else '暂无'}",
                    f"- 关注价位：{plan.get('focus_price', item['next_focus'])}",
                    f"- 失效条件：{plan.get('invalidation', '暂无')}",
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
    _log(f"render_pipeline_daily_report wrote path={output_path} chars={len(''.join(body))}")


def render_pipeline_plan(plan_output: PlanAgentOutput, output_path: Path) -> None:
    _log(f"render_pipeline_plan start path={output_path} symbols={len(plan_output.symbols)}")
    groups: dict[str, list[dict[str, Any]]] = {action: [] for action in PLAN_ACTIONS}
    for item in plan_output.symbols:
        groups.setdefault(item.final_action, []).append(item.model_dump())

    lines = [f"# 次日交易计划（{plan_output.review_date}）", "", "## 总览"]
    lines.append(f"- 市场概览：{plan_output.market_summary}")
    lines.append(f"- 组合视角：{plan_output.portfolio_summary}")
    lines.append(f"- 核心安排：{plan_output.next_day_overview}")
    if plan_output.warnings:
        lines.append(f"- 运行提示：{'；'.join(plan_output.warnings)}")
    lines.append("")

    for action in PLAN_ACTIONS:
        lines.append(f"## {action}")
        items = groups.get(action) or []
        if not items:
            lines.extend(["- 暂无", ""])
            continue
        for item in items:
            lines.extend(
                [
                    f"### {item['symbol']} {item['name']}",
                    f"- 计划动作：{item['final_action']}",
                    f"- 核心结论：{item['thesis']}",
                    f"- 执行步骤：{'；'.join(item.get('plan_steps', []))}",
                    f"- 关注价位：{item['focus_price']}",
                    f"- 失效条件：{item['invalidation']}",
                    f"- 风险等级：{item['risk_level']}",
                    f"- 单标的仓位上限：{item['position_pct_limit'] * 100:.1f}%",
                    f"- 风险提示：{'；'.join(item.get('risk_flags', [])) if item.get('risk_flags') else '暂无'}",
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"render_pipeline_plan wrote path={output_path} lines={len(lines)}")


def write_pipeline_json(
    data_output: DataAgentOutput,
    research_output: ResearchAgentOutput,
    risk_output: RiskAgentOutput,
    plan_output: PlanAgentOutput,
    output_path: Path,
) -> None:
    _log(f"write_pipeline_json start path={output_path}")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_agent": data_output.model_dump(),
        "research_agent": research_output.model_dump(),
        "risk_agent": risk_output.model_dump(),
        "plan_agent": plan_output.model_dump(),
        "results": merge_pipeline_results(data_output, research_output, risk_output, plan_output),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path.write_text(content, encoding="utf-8")
    _log(f"write_pipeline_json wrote path={output_path} chars={len(content)}")
