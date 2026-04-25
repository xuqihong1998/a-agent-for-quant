from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tools.data_loader import load_symbol_df, load_watchlist
from tools.indicators import calc_basic_indicators
from tools.portfolio import load_positions, positions_to_map
from tools.report_writer import enrich_position_info, summarize_symbol
from tools.risk_checks import classify_plan_action


PLAN_ACTIONS = ["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_review_targets(
    watchlist: list[dict[str, Any]],
    position_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {item["symbol"]: dict(item) for item in watchlist}
    for symbol, position in position_map.items():
        if symbol not in merged:
            merged[symbol] = {"symbol": symbol, "name": position.get("name", symbol)}
    return sorted(merged.values(), key=lambda item: item["symbol"])


def build_review_results(config_dir: Path, data_dir: Path) -> list[dict[str, Any]]:
    watchlist = load_watchlist(config_dir)
    positions = load_positions(config_dir / "positions.yaml")
    position_map = positions_to_map(positions)
    review_targets = merge_review_targets(watchlist, position_map)
    results: list[dict[str, Any]] = []

    for item in review_targets:
        symbol = item["symbol"]
        name = item["name"]
        try:
            df = load_symbol_df(data_dir, symbol)
            df = calc_basic_indicators(df)
            summary = summarize_symbol(symbol, df, name=name)

            if symbol in position_map:
                summary = enrich_position_info(summary, position_map[symbol])
            else:
                summary["is_position"] = False

            results.append(summary)
        except Exception:
            continue

    return results


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def serialize_summary(item: dict[str, Any]) -> dict[str, Any]:
    serialized = {key: _serialize_value(value) for key, value in item.items()}
    risk_notes = serialized.get("risk_notes") or []
    serialized["risk_notes"] = [str(note) for note in risk_notes]
    return serialized


def build_review_context(
    config_dir: Path,
    data_dir: Path,
    review_date: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = build_review_results(config_dir, data_dir)
    positions = load_positions(config_dir / "positions.yaml")
    watchlist = load_watchlist(config_dir)
    today = review_date or datetime.now().strftime("%Y-%m-%d")

    context = {
        "review_date": today,
        "watchlist": [_serialize_value(item) for item in watchlist],
        "positions": [_serialize_value(item) for item in positions],
        "risk_rules": _serialize_value(_read_yaml(config_dir / "risk_rules.yaml")),
        "summary_items": [serialize_summary(item) for item in results],
        "summary_stats": {
            "position_count": sum(1 for item in results if item.get("is_position")),
            "watch_count": sum(1 for item in results if not item.get("is_position")),
            "up_count": sum(1 for item in results if item.get("trend") == "up"),
            "neutral_count": sum(1 for item in results if item.get("trend") == "neutral"),
            "weak_count": sum(1 for item in results if item.get("trend") == "weak"),
        },
    }
    return context, results


def merge_agent_output(
    base_results: list[dict[str, Any]],
    agent_output: dict[str, Any],
) -> list[dict[str, Any]]:
    symbol_reviews = {
        str(item["symbol"]).zfill(6): item for item in agent_output.get("symbols", []) if item.get("symbol")
    }
    merged_results: list[dict[str, Any]] = []

    for item in base_results:
        merged = dict(item)
        review = symbol_reviews.get(item["symbol"], {})
        merged["agent_headline"] = review.get("headline", item["comment"])
        merged["agent_thesis"] = review.get("thesis", item["comment"])
        merged["agent_key_signals"] = review.get("key_signals", [])
        merged["agent_risk_flags"] = review.get("risk_flags", item.get("risk_notes", []))
        merged["agent_action"] = review.get(
            "stance",
            item.get("position_action") if item.get("is_position") else classify_plan_action(item),
        )
        merged["agent_next_day_plan"] = review.get("next_day_plan", item["action_hint"])
        merged["agent_focus_price"] = review.get("focus_price", item["next_focus"])
        merged["agent_invalidation"] = review.get("invalidation", "关注既有风险规则是否触发。")
        merged["agent_confidence"] = review.get("confidence", 0.5)
        merged_results.append(merged)

    return merged_results


def _format_ai_review_lines(item: dict[str, Any]) -> list[str]:
    signal_text = "；".join(item.get("agent_key_signals", [])) if item.get("agent_key_signals") else "暂无补充信号"
    risk_text = "；".join(item.get("agent_risk_flags", [])) if item.get("agent_risk_flags") else "暂无新增风险提示"
    return [
        f"- AI 研究标题：{item['agent_headline']}",
        f"- AI 研究结论：{item['agent_thesis']}",
        f"- AI 关键信号：{signal_text}",
        f"- AI 次日计划：{item['agent_next_day_plan']}",
        f"- AI 关注价位：{item['agent_focus_price']}",
        f"- AI 失效条件：{item['agent_invalidation']}",
        f"- AI 风险提示：{risk_text}",
        f"- AI 置信度：{item['agent_confidence']:.2f}",
        "",
    ]


def _format_symbol_section(item: dict[str, Any]) -> list[str]:
    title = item["symbol"] if item["name"] == item["symbol"] else f"{item['symbol']} {item['name']}"
    lines = [
        f"### {title}",
        f"- 收盘价：{item['close']:.2f}",
        f"- 涨跌幅：{item['pct_chg'] * 100:.2f}%",
        f"- MA5 / MA10 / MA20：{item['ma5']:.2f} / {item['ma10']:.2f} / {item['ma20']:.2f}",
        f"- ATR14：{item['atr14']:.2f}",
        f"- 量能状态：{item['volume_status']} ({item['amount_ratio_5d']:.2f})",
        f"- 趋势判断：{item['trend']}",
        f"- 规则结论：{item['comment']}",
        f"- 原始观察点：{item['next_focus']}",
    ]
    lines.extend(_format_ai_review_lines(item))
    return lines


def _format_position_section(item: dict[str, Any]) -> list[str]:
    title = item["symbol"] if item.get("position_name") == item["symbol"] else f"{item['symbol']} {item.get('position_name', item['name'])}"
    risk_line = "；".join(item.get("risk_notes", [])) if item.get("risk_notes") else "暂无明显风险信号"
    lines = [
        f"### {title}",
        f"- 持仓动作：{item.get('position_action', '持有')}",
        f"- 成本价：{item['cost']:.2f}",
        f"- 当前价：{item['close']:.2f}",
        f"- 持股数量：{item['shares']}",
        f"- 持仓市值：{item['market_value']:.2f}",
        f"- 浮盈亏 %：{item['pnl_pct'] * 100:.2f}%",
        f"- 浮盈亏金额：{item['pnl_amount']:.2f}",
        f"- MA5 / MA10 / MA20：{item['ma5']:.2f} / {item['ma10']:.2f} / {item['ma20']:.2f}",
        f"- 量能状态：{item['volume_status']} ({item['amount_ratio_5d']:.2f})",
        f"- 规则风险提示：{risk_line}",
        f"- 原始动作建议：{item['action_hint']}",
    ]
    lines.extend(_format_ai_review_lines(item))
    return lines


def render_agent_daily_report(
    base_results: list[dict[str, Any]],
    agent_output: dict[str, Any],
    output_path: Path,
    title: str = "盘后复盘报告",
) -> None:
    merged_results = merge_agent_output(base_results, agent_output)
    positions = [item for item in merged_results if item.get("is_position")]
    watch_only = [item for item in merged_results if not item.get("is_position")]
    ups = [item for item in merged_results if item["trend"] == "up"]
    neutrals = [item for item in merged_results if item["trend"] == "neutral"]
    weaks = [item for item in merged_results if item["trend"] == "weak"]
    report_date = output_path.stem if output_path.stem else datetime.now().strftime("%Y-%m-%d")

    body = [
        f"# {title}（{report_date}）",
        "",
        "## 总览",
        f"- 趋势向上：{len(ups)}",
        f"- 趋势中性：{len(neutrals)}",
        f"- 趋势转弱：{len(weaks)}",
        f"- 运行模式：{agent_output.get('run_mode', 'unknown')}",
        "",
        "## AI 总结",
        f"- 市场概览：{agent_output.get('market_summary', '暂无')} ",
        f"- 持仓视角：{agent_output.get('portfolio_summary', '暂无')}",
        f"- 次日总计划：{agent_output.get('next_day_overview', '暂无')}",
    ]

    warnings = agent_output.get("warnings", [])
    if warnings:
        body.append(f"- 运行提示：{'；'.join(str(item) for item in warnings)}")
    body.append("")

    body.append("## 持仓分析")
    if positions:
        for item in positions:
            body.extend(_format_position_section(item))
    else:
        body.extend(["- 当前无持仓分析对象", ""])

    body.append("## 观察池分析")
    if watch_only:
        for item in watch_only:
            body.extend(_format_symbol_section(item))
    else:
        body.extend(["- 当前无观察池标的", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")


def write_agent_review_json(
    base_results: list[dict[str, Any]],
    agent_output: dict[str, Any],
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "review": agent_output,
        "results": [serialize_summary(item) for item in merge_agent_output(base_results, agent_output)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
