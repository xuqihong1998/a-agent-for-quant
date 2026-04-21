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


def summarize_symbol(symbol: str, df: pd.DataFrame, name: str | None = None) -> dict[str, Any]:
    if df.empty:
        raise ValueError(f"{symbol} has no rows")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    close = _safe_float(last["close"])
    prev_close = _safe_float(prev["close"], close)
    pct_chg = (close - prev_close) / prev_close if prev_close else 0.0

    ma5 = _safe_float(last.get("ma5"))
    ma10 = _safe_float(last.get("ma10"))
    ma20 = _safe_float(last.get("ma20"))
    atr14 = _safe_float(last.get("atr14"))
    amount_ratio_5d = _safe_float(last.get("amount_ratio_5d"), 1.0)

    above_ma5 = close >= ma5 if ma5 else False
    above_ma10 = close >= ma10 if ma10 else False
    above_ma20 = close >= ma20 if ma20 else False

    if above_ma5 and above_ma10 and above_ma20:
        trend = "up"
    elif (not above_ma10) and (not above_ma20):
        trend = "weak"
    else:
        trend = "neutral"

    if amount_ratio_5d >= 1.2:
        volume_status = "放量"
    elif amount_ratio_5d <= 0.8:
        volume_status = "缩量"
    else:
        volume_status = "平量"

    if trend == "up":
        comment = "趋势保持，优先看 MA5/MA10 附近承接。"
    elif trend == "weak":
        comment = "趋势转弱，重点看是否跌破 MA20 后继续走低。"
    else:
        comment = "趋势中性，等待方向进一步确认。"

    return {
        "symbol": symbol,
        "name": name or symbol,
        "close": close,
        "pct_chg": pct_chg,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "atr14": atr14,
        "amount_ratio_5d": amount_ratio_5d,
        "trend": trend,
        "above_ma5": above_ma5,
        "above_ma10": above_ma10,
        "above_ma20": above_ma20,
        "volume_status": volume_status,
        "comment": comment,
    }


def enrich_position_info(item: dict[str, Any], pos: dict[str, Any]) -> dict[str, Any]:
    cost = float(pos["cost"])
    shares = int(pos["shares"])

    item["is_position"] = True
    item["position_name"] = pos.get("name", item["symbol"])
    item["cost"] = cost
    item["shares"] = shares
    item["market_value"] = item["close"] * shares
    item["pnl_pct"] = (item["close"] - cost) / cost if cost else 0.0
    item["pnl_amount"] = (item["close"] - cost) * shares

    risk_notes: list[str] = []
    if item["ma10"] and item["close"] < item["ma10"]:
        risk_notes.append("跌破 MA10")
    if item["ma20"] and item["close"] < item["ma20"]:
        risk_notes.append("跌破 MA20")
    if item["pnl_pct"] <= -0.06:
        risk_notes.append("接近/触发固定止损")
    if item["amount_ratio_5d"] >= 1.5 and item["pct_chg"] < 0:
        risk_notes.append("放量下跌")

    item["risk_notes"] = risk_notes
    return item


def format_symbol_markdown(item: dict[str, Any]) -> str:
    title = item["symbol"] if item["name"] == item["symbol"] else f"{item['symbol']} {item['name']}"
    return f"""### {title}
- 收盘价：{item['close']:.2f}
- 涨跌幅：{item['pct_chg'] * 100:.2f}%
- MA5 / MA10 / MA20：{item['ma5']:.2f} / {item['ma10']:.2f} / {item['ma20']:.2f}
- ATR14：{item['atr14']:.2f}
- 量能状态：{item['volume_status']}（{item['amount_ratio_5d']:.2f}）
- 趋势判断：{item['trend']}
- 复盘结论：{item['comment']}
"""


def format_position_markdown(item: dict[str, Any]) -> str:
    risk_line = "、".join(item.get("risk_notes", [])) if item.get("risk_notes") else "无明显异常"
    title = item["symbol"] if not item.get("position_name") else f"{item['symbol']} {item['position_name']}"
    return f"""### {title}
- 成本价：{item['cost']:.2f}
- 当前价：{item['close']:.2f}
- 持股数量：{item['shares']}
- 持仓市值：{item['market_value']:.2f}
- 浮盈亏：{item['pnl_pct'] * 100:.2f}% / {item['pnl_amount']:.2f}
- MA5 / MA10 / MA20：{item['ma5']:.2f} / {item['ma10']:.2f} / {item['ma20']:.2f}
- 量能状态：{item['volume_status']}（{item['amount_ratio_5d']:.2f}）
- 风险提示：{risk_line}
- 复盘结论：{item['comment']}
"""


def render_daily_report(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str = "盘后复盘报告",
) -> None:
    report_date = output_path.stem if output_path.stem else datetime.now().strftime("%Y-%m-%d")
    body = [f"# {title}（{report_date}）\n"]

    positions = [x for x in results if x.get("is_position")]
    watch_only = [x for x in results if not x.get("is_position")]
    ups = [x for x in results if x["trend"] == "up"]
    weaks = [x for x in results if x["trend"] == "weak"]
    neutrals = [x for x in results if x["trend"] == "neutral"]

    body.append("## 总览")
    body.append(f"- 股票数量：{len(results)}")
    body.append(f"- 持仓股：{len(positions)}")
    body.append(f"- 观察股：{len(watch_only)}")
    body.append(f"- 趋势向上：{len(ups)}")
    body.append(f"- 趋势中性：{len(neutrals)}")
    body.append(f"- 趋势转弱：{len(weaks)}\n")

    body.append("## 持仓分析")
    if positions:
        for item in positions:
            body.append(format_position_markdown(item))
    else:
        body.append("- 当前无持仓\n")

    body.append("## 观察池分析")
    if watch_only:
        for item in watch_only:
            body.append(format_symbol_markdown(item))
    else:
        body.append("- 当前无观察股\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
