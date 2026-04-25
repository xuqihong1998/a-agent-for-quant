from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tools.data_loader import load_symbol_df, load_watchlist
from tools.indicators import calc_basic_indicators
from tools.portfolio import load_positions, positions_to_map
from tools.report_writer import enrich_position_info, summarize_symbol


class DataSymbolSummary(BaseModel):
    symbol: str
    name: str
    category: Literal["position", "watchlist"]
    tags: list[str] = Field(default_factory=list)
    date: str | None = None
    is_position: bool
    close: float
    pct_chg: float
    ma5: float
    ma10: float
    ma20: float
    atr14: float
    amount_ratio_5d: float
    volume_status: str
    trend: str
    comment: str
    next_focus: str
    action_hint: str
    high: float | None = None
    support_level: float | None = None
    pressure_level: float | None = None
    position_action: str | None = None
    cost: float | None = None
    shares: int | None = None
    market_value: float | None = None
    pnl_pct: float | None = None
    pnl_amount: float | None = None
    risk_notes: list[str] = Field(default_factory=list)


class DataAgentOutput(BaseModel):
    review_date: str
    run_mode: Literal["deterministic_data_agent"]
    summary_stats: dict[str, int]
    symbols: list[DataSymbolSummary]
    warnings: list[str] = Field(default_factory=list)


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _normalize_data_summary(item: dict[str, Any], tags: list[str]) -> DataSymbolSummary:
    return DataSymbolSummary.model_validate(
        {
            "symbol": item["symbol"],
            "name": item["name"],
            "category": "position" if item.get("is_position") else "watchlist",
            "tags": tags,
            "date": item.get("date").strftime("%Y-%m-%d") if item.get("date") is not None else None,
            "is_position": bool(item.get("is_position")),
            "close": float(item["close"]),
            "pct_chg": float(item["pct_chg"]),
            "ma5": float(item["ma5"]),
            "ma10": float(item["ma10"]),
            "ma20": float(item["ma20"]),
            "atr14": float(item["atr14"]),
            "amount_ratio_5d": float(item["amount_ratio_5d"]),
            "volume_status": str(item["volume_status"]),
            "trend": str(item["trend"]),
            "comment": str(item["comment"]),
            "next_focus": str(item["next_focus"]),
            "action_hint": str(item["action_hint"]),
            "high": float(item["high"]) if item.get("high") is not None else None,
            "support_level": float(item["support_level"]) if item.get("support_level") is not None else None,
            "pressure_level": float(item["pressure_level"]) if item.get("pressure_level") is not None else None,
            "position_action": str(item["position_action"]) if item.get("position_action") else None,
            "cost": float(item["cost"]) if item.get("cost") is not None else None,
            "shares": int(item["shares"]) if item.get("shares") is not None else None,
            "market_value": float(item["market_value"]) if item.get("market_value") is not None else None,
            "pnl_pct": float(item["pnl_pct"]) if item.get("pnl_pct") is not None else None,
            "pnl_amount": float(item["pnl_amount"]) if item.get("pnl_amount") is not None else None,
            "risk_notes": _safe_list(item.get("risk_notes")),
        }
    )


def run_data_agent(
    config_dir: Path,
    data_dir: Path,
    review_date: str | None = None,
) -> DataAgentOutput:
    watchlist = load_watchlist(config_dir)
    watch_tags = {item["symbol"]: _safe_list(item.get("tags")) for item in watchlist}
    positions = load_positions(config_dir / "positions.yaml")
    position_map = positions_to_map(positions)

    targets: dict[str, dict[str, Any]] = {item["symbol"]: dict(item) for item in watchlist}
    for symbol, position in position_map.items():
        if symbol not in targets:
            targets[symbol] = {"symbol": symbol, "name": position.get("name", symbol)}

    summaries: list[DataSymbolSummary] = []
    warnings: list[str] = []

    for symbol in sorted(targets):
        item = targets[symbol]
        try:
            df = load_symbol_df(data_dir, symbol)
            df = calc_basic_indicators(df)
            summary = summarize_symbol(symbol, df, name=item["name"])
            if symbol in position_map:
                summary = enrich_position_info(summary, position_map[symbol])
            else:
                summary["is_position"] = False
            summaries.append(_normalize_data_summary(summary, tags=watch_tags.get(symbol, [])))
        except Exception as exc:
            warnings.append(f"{symbol} 数据处理失败: {exc}")

    resolved_review_date = review_date or datetime.now().strftime("%Y-%m-%d")
    summary_stats = {
        "position_count": sum(1 for item in summaries if item.is_position),
        "watch_count": sum(1 for item in summaries if not item.is_position),
        "up_count": sum(1 for item in summaries if item.trend == "up"),
        "neutral_count": sum(1 for item in summaries if item.trend == "neutral"),
        "weak_count": sum(1 for item in summaries if item.trend == "weak"),
    }
    return DataAgentOutput(
        review_date=resolved_review_date,
        run_mode="deterministic_data_agent",
        summary_stats=summary_stats,
        symbols=summaries,
        warnings=warnings,
    )
