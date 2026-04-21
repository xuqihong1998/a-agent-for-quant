from __future__ import annotations

from typing import Any


def classify_plan_action(item: dict[str, Any]) -> str:
    """Classify tomorrow's action from review summary data."""
    if item.get("is_position"):
        if item["close"] < item["ma20"]:
            return "减仓观察"
        if item["close"] < item["ma10"]:
            return "谨慎持有"
        return "继续持有"

    if item["trend"] == "up" and item["amount_ratio_5d"] >= 1.2:
        return "重点观察"
    if item["trend"] == "neutral":
        return "普通观察"
    return "暂不关注"


def generate_symbol_plan(item: dict[str, Any]) -> dict[str, Any]:
    action = classify_plan_action(item)

    ma5 = item["ma5"]
    ma10 = item["ma10"]
    ma20 = item["ma20"]
    focus_low = min(ma5, ma10)
    focus_high = max(ma5, ma10)

    if action in ("继续持有", "谨慎持有"):
        entry_condition = "观察 MA5/MA10 附近承接，不急于追高"
        invalid_condition = "有效跌破 MA20 或放量长阴"
        focus_price = f"关注 {focus_low:.2f} ~ {focus_high:.2f}"
    elif action == "重点观察":
        entry_condition = "回踩不破 MA5/MA10 后重新放量上行"
        invalid_condition = "跌破 MA10 且量价走弱"
        focus_price = f"关注 {focus_low:.2f} ~ {focus_high:.2f}"
    elif action == "减仓观察":
        entry_condition = "除非快速收复 MA10，否则不主动加仓"
        invalid_condition = "继续跌破 MA20 并放量走弱"
        focus_price = f"重点看 MA20={ma20:.2f}"
    else:
        entry_condition = "暂无明确入场点"
        invalid_condition = "无需定义"
        focus_price = "不重点跟踪"

    risk_note = "、".join(item.get("risk_notes", [])) if item.get("risk_notes") else "常规观察"

    return {
        "symbol": item["symbol"],
        "name": item.get("position_name") or item.get("name", item["symbol"]),
        "action": action,
        "focus_price": focus_price,
        "entry_condition": entry_condition,
        "invalid_condition": invalid_condition,
        "risk_note": risk_note,
    }
