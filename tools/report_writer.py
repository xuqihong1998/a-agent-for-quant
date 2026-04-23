from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return default
    return float(number)


def _pick_latest_valid_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("symbol dataframe is empty")

    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result = result.sort_values("date").reset_index(drop=True)

    required = ["close", "ma5", "ma10", "ma20", "atr14", "amount_ratio_5d", "pct_chg"]
    available_required = [column for column in required if column in result.columns]
    if available_required:
        valid_rows = result.dropna(subset=available_required, how="any")
        if not valid_rows.empty:
            return valid_rows.iloc[-1]

    return result.iloc[-1]


def judge_trend(last_row: pd.Series) -> str:
    close = _safe_float(last_row.get("close"))
    ma5 = _safe_float(last_row.get("ma5"))
    ma10 = _safe_float(last_row.get("ma10"))
    ma20 = _safe_float(last_row.get("ma20"))

    if close >= ma5 and close >= ma10 and close >= ma20:
        return "up"
    if close < ma10 and close < ma20:
        return "weak"
    return "neutral"


def judge_volume_status(last_row: pd.Series) -> str:
    amount_ratio = _safe_float(last_row.get("amount_ratio_5d"), default=1.0)
    if amount_ratio >= 1.2:
        return "放量"
    if amount_ratio <= 0.8:
        return "缩量"
    return "平量"


def build_symbol_comment(summary: dict[str, Any]) -> str:
    comment_map = {
        ("up", "放量"): "趋势保持，关注强势延续",
        ("up", "缩量"): "趋势未坏，但量能一般",
        ("up", "平量"): "趋势保持，继续观察均线支撑",
        ("weak", "放量"): "警惕放量走弱",
        ("weak", "缩量"): "弱势整理，优先观察支撑",
        ("weak", "平量"): "走势偏弱，先看止跌信号",
        ("neutral", "放量"): "量能放大但趋势未明，留意方向选择",
        ("neutral", "缩量"): "量能偏弱，耐心等待突破",
        ("neutral", "平量"): "趋势中性，继续观察价格与均线关系",
    }
    return comment_map.get(
        (summary["trend"], summary["volume_status"]),
        "趋势中性，继续观察后续变化",
    )


def _build_next_day_plan(summary: dict[str, Any]) -> dict[str, Any]:
    trend = summary["trend"]
    ma5 = summary["ma5"]
    ma10 = summary["ma10"]
    ma20 = summary["ma20"]
    high = summary["high"]

    if trend == "up":
        return {
            "next_focus": f"关注 MA5={ma5:.2f} 与 MA10={ma10:.2f} 附近承接",
            "support_level": ma10,
            "pressure_level": high,
            "action_hint": "回踩不破可继续关注",
        }
    if trend == "neutral":
        return {
            "next_focus": f"观察能否站稳 MA10={ma10:.2f}",
            "support_level": ma10,
            "pressure_level": high,
            "action_hint": "先观察，不急于出手",
        }
    return {
        "next_focus": f"留意 MA20={ma20:.2f} 是否失守",
        "support_level": ma20,
        "pressure_level": ma10,
        "action_hint": "跌破关键位则降级观察",
    }


def summarize_symbol(symbol: str, df: pd.DataFrame, name: str | None = None) -> dict[str, Any]:
    last_row = _pick_latest_valid_row(df)

    close = _safe_float(last_row.get("close"))
    ma5 = _safe_float(last_row.get("ma5"))
    ma10 = _safe_float(last_row.get("ma10"))
    ma20 = _safe_float(last_row.get("ma20"))

    summary = {
        "symbol": str(symbol).zfill(6),
        "name": name or str(last_row.get("name") or str(symbol).zfill(6)),
        "date": pd.to_datetime(last_row.get("date"), errors="coerce") if "date" in last_row.index else pd.NaT,
        "close": close,
        "high": _safe_float(last_row.get("high"), default=close),
        "pct_chg": _safe_float(last_row.get("pct_chg")),
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "atr14": _safe_float(last_row.get("atr14")),
        "amount_ratio_5d": _safe_float(last_row.get("amount_ratio_5d"), default=1.0),
    }
    summary["trend"] = judge_trend(last_row)
    summary["volume_status"] = judge_volume_status(last_row)
    summary["above_ma5"] = close >= ma5 if ma5 else False
    summary["above_ma10"] = close >= ma10 if ma10 else False
    summary["above_ma20"] = close >= ma20 if ma20 else False
    summary["comment"] = build_symbol_comment(summary)
    summary.update(_build_next_day_plan(summary))
    return summary


def enrich_position_info(item: dict[str, Any], position: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    cost = _safe_float(position.get("cost"))
    shares = int(pd.to_numeric(position.get("shares"), errors="coerce"))

    enriched["is_position"] = True
    enriched["position_name"] = str(position.get("name") or enriched["name"])
    enriched["cost"] = cost
    enriched["shares"] = shares
    enriched["market_value"] = enriched["close"] * shares
    enriched["pnl_pct"] = (enriched["close"] - cost) / cost if cost else 0.0
    enriched["pnl_amount"] = (enriched["close"] - cost) * shares

    risk_notes: list[str] = []
    if enriched["ma10"] and enriched["close"] < enriched["ma10"]:
        risk_notes.append("跌破MA10，短线转弱")
    if enriched["ma20"] and enriched["close"] < enriched["ma20"]:
        risk_notes.append("跌破MA20，中期趋势受压")
    if enriched["pnl_pct"] <= -0.06:
        risk_notes.append("浮亏接近固定止损")
    if enriched["amount_ratio_5d"] >= 1.2 and enriched["pct_chg"] < 0:
        risk_notes.append("放量下跌，警惕资金流出")

    trend = enriched["trend"]
    if trend == "up":
        position_action = "持有"
        enriched["action_hint"] = "持有，回踩关键均线不破可继续跟踪"
    elif trend == "neutral":
        position_action = "谨慎持有"
        enriched["action_hint"] = "谨慎持有，先看能否重新站稳MA10"
    else:
        position_action = "减仓观察"
        enriched["action_hint"] = "减仓观察，跌破关键位则进一步降级处理"

    enriched["position_action"] = position_action
    enriched["risk_notes"] = risk_notes or ["暂无明显风险信号"]
    return enriched


def format_symbol_markdown(item: dict[str, Any]) -> str:
    title = item["symbol"] if item["name"] == item["symbol"] else f"{item['symbol']} {item['name']}"
    return "\n".join(
        [
            f"### {title}",
            f"- 收盘价：{item['close']:.2f}",
            f"- 涨跌幅：{item['pct_chg'] * 100:.2f}%",
            f"- MA5 / MA10 / MA20：{item['ma5']:.2f} / {item['ma10']:.2f} / {item['ma20']:.2f}",
            f"- ATR14：{item['atr14']:.2f}",
            f"- 量能状态：{item['volume_status']} ({item['amount_ratio_5d']:.2f})",
            f"- 趋势判断：{item['trend']}",
            f"- 复盘结论：{item['comment']}",
            f"- 明日观察点：{item['next_focus']}",
            f"- 动作建议：{item['action_hint']}",
            "",
        ]
    )


def format_position_markdown(item: dict[str, Any]) -> str:
    title = item["symbol"] if item.get("position_name") == item["symbol"] else f"{item['symbol']} {item.get('position_name', item['name'])}"
    risk_line = "；".join(item.get("risk_notes", [])) if item.get("risk_notes") else "暂无明显风险信号"
    return "\n".join(
        [
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
            f"- 风险提示：{risk_line}",
            f"- 复盘结论：{item['comment']}",
            f"- 明日观察点：{item['next_focus']}",
            f"- 动作建议：{item['action_hint']}",
            "",
        ]
    )


def render_daily_report(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str = "盘后复盘报告",
) -> None:
    positions = [item for item in results if item.get("is_position")]
    watch_only = [item for item in results if not item.get("is_position")]
    ups = [item for item in results if item["trend"] == "up"]
    neutrals = [item for item in results if item["trend"] == "neutral"]
    weaks = [item for item in results if item["trend"] == "weak"]
    report_date = output_path.stem if output_path.stem else datetime.now().strftime("%Y-%m-%d")

    body = [
        f"# {title}（{report_date}）",
        "",
        "## 总览",
        f"- 趋势向上：{len(ups)}",
        f"- 趋势中性：{len(neutrals)}",
        f"- 趋势转弱：{len(weaks)}",
        "",
        "## 持仓分析",
    ]

    if positions:
        for item in positions:
            body.append(format_position_markdown(item))
    else:
        body.extend(["- 当前无持仓分析对象", ""])

    body.append("## 观察池分析")
    if watch_only:
        for item in watch_only:
            body.append(format_symbol_markdown(item))
    else:
        body.extend(["- 当前无观察池标的", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
