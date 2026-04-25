from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tools.market_data import load_symbol_df


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
REPORT_DIR = PROJECT_ROOT / "reports" / "backtests"


ACTION_TO_TARGET_PCT = {
    "继续持有": 0.20,
    "谨慎持有": 0.10,
    "重点观察": 0.08,
    "普通观察": 0.00,
    "减仓观察": 0.00,
    "暂不关注": 0.00,
}


@dataclass
class SignalSnapshot:
    review_date: str
    symbols: list[dict[str, Any]]
    source_path: Path


def _log(message: str) -> None:
    print(f"[BACKTEST] {message}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_signal_snapshot(path: Path) -> SignalSnapshot:
    _log(f"load_signal_snapshot path={path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan_agent = payload.get("plan_agent") or {}
    symbols = plan_agent.get("symbols") or []
    review_date = str(plan_agent.get("review_date") or payload.get("generated_at", ""))[:10]
    snapshot = SignalSnapshot(review_date=review_date, symbols=symbols, source_path=path)
    _log(f"loaded signal snapshot review_date={snapshot.review_date} symbols={len(snapshot.symbols)}")
    return snapshot


def signal_to_trade_plan(symbol_signal: dict[str, Any]) -> dict[str, Any]:
    action = str(symbol_signal.get("final_action") or "普通观察")
    risk_level = str(symbol_signal.get("risk_level") or "medium")
    base_target_pct = float(symbol_signal.get("position_pct_limit", ACTION_TO_TARGET_PCT.get(action, 0.0)))
    if risk_level == "high":
        target_pct = min(base_target_pct, 0.05)
    elif risk_level == "medium":
        target_pct = min(base_target_pct, 0.10)
    else:
        target_pct = base_target_pct

    plan = {
        "symbol": str(symbol_signal["symbol"]).zfill(6),
        "name": str(symbol_signal.get("name") or symbol_signal["symbol"]),
        "action": action,
        "risk_level": risk_level,
        "target_pct": max(target_pct, 0.0),
        "focus_price": str(symbol_signal.get("focus_price") or ""),
        "invalidation": str(symbol_signal.get("invalidation") or ""),
        "confidence": float(symbol_signal.get("confidence", 0.5)),
        "thesis": str(symbol_signal.get("thesis") or ""),
        "plan_steps": [str(item) for item in (symbol_signal.get("plan_steps") or [])],
        "risk_flags": [str(item) for item in (symbol_signal.get("risk_flags") or [])],
    }
    return plan


def build_trade_plan_from_snapshot(snapshot: SignalSnapshot) -> dict[str, dict[str, Any]]:
    trade_plan = {
        item["symbol"]: signal_to_trade_plan(item)
        for item in snapshot.symbols
        if item.get("symbol")
    }
    _log(f"built trade plan symbols={len(trade_plan)}")
    return trade_plan


def build_price_frame(
    symbol: str,
    data_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = load_symbol_df(symbol, data_dir=data_dir).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_signal_feed_frames(
    trade_plan: dict[str, dict[str, Any]],
    data_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, plan in trade_plan.items():
        df = build_price_frame(symbol, data_dir=data_dir, start_date=start_date, end_date=end_date)
        if df.empty:
            _log(f"skip empty frame symbol={symbol}")
            continue

        df["signal_target_pct"] = plan["target_pct"]
        df["signal_confidence"] = plan["confidence"]
        df["signal_action"] = plan["action"]
        df["signal_risk_level"] = plan["risk_level"]
        frames[symbol] = df
    _log(f"built signal feed frames symbols={len(frames)}")
    return frames


def load_backtest_settings(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    settings = _read_yaml(config_dir / "settings.yaml")
    risk_rules = _read_yaml(config_dir / "risk_rules.yaml")
    llm_settings = settings.get("llm", {})
    trade_rules = risk_rules.get("trade", {})
    portfolio_rules = risk_rules.get("portfolio", {})
    return {
        "initial_cash": float(settings.get("backtest", {}).get("initial_cash", 1_000_000)),
        "commission": float(settings.get("backtest", {}).get("commission", 0.001)),
        "slippage_perc": float(settings.get("backtest", {}).get("slippage_perc", 0.0005)),
        "stop_loss_pct": float(trade_rules.get("stop_loss_pct", 0.06)),
        "take_profit_pct": float(trade_rules.get("take_profit_pct", 0.15)),
        "trailing_stop_pct": float(trade_rules.get("trailing_stop_pct", 0.08)),
        "min_cash_pct": float(trade_rules.get("min_cash_pct", 0.2)),
        "max_position_pct": float(portfolio_rules.get("max_position_pct", 0.2)),
        "model_name": llm_settings.get("model"),
    }


def summarize_backtest_result(
    snapshot: SignalSnapshot,
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_source": str(snapshot.source_path),
        "review_date": snapshot.review_date,
        "metrics": metrics,
        "trades": trades,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"summarize_backtest_result wrote path={output_path}")


def render_backtest_markdown(
    snapshot: SignalSnapshot,
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        f"# Agent 信号回测报告（{snapshot.review_date}）",
        "",
        "## 总览",
        f"- 信号来源：{snapshot.source_path}",
        f"- 期末净值：{metrics['final_value']:.2f}",
        f"- 总收益率：{metrics['total_return_pct'] * 100:.2f}%",
        f"- 交易次数：{metrics['trade_count']}",
        f"- 胜率：{metrics['win_rate'] * 100:.2f}%",
        "",
        "## 最近交易",
    ]

    if not trades:
        lines.extend(["- 暂无交易", ""])
    else:
        for trade in trades[-10:]:
            lines.extend(
                [
                    f"### {trade['symbol']} {trade['name']}",
                    f"- 开仓日期：{trade['opened_at']}",
                    f"- 平仓日期：{trade['closed_at']}",
                    f"- 收益率：{trade['pnl_pct'] * 100:.2f}%",
                    f"- 收益金额：{trade['pnl_amount']:.2f}",
                    f"- 退出原因：{trade['exit_reason']}",
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"render_backtest_markdown wrote path={output_path}")
