from __future__ import annotations

import argparse
import sys
from pathlib import Path

import backtrader as bt


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from strategies.agent_signal_strategy import AgentSignalStrategy, SignalPandasData
from tools.backtest import (
    REPORT_DIR,
    build_signal_feed_frames,
    build_trade_plan_from_snapshot,
    load_backtest_settings,
    load_signal_snapshot,
    render_backtest_markdown,
    summarize_backtest_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest agent-generated signals with Backtrader.")
    parser.add_argument(
        "--signal-json",
        required=True,
        help="Path to agent signal json, e.g. reports/daily/2026-04-25.pipeline.json",
    )
    parser.add_argument("--start", default=None, help="Optional backtest start date, e.g. 2026-01-01")
    parser.add_argument("--end", default=None, help="Optional backtest end date, e.g. 2026-04-25")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data" / "raw"), help="Raw market data directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal_path = Path(args.signal_json).resolve()
    data_dir = Path(args.data_dir).resolve()

    snapshot = load_signal_snapshot(signal_path)
    trade_plan = build_trade_plan_from_snapshot(snapshot)
    frames = build_signal_feed_frames(trade_plan, data_dir=data_dir, start_date=args.start, end_date=args.end)
    if not frames:
        raise RuntimeError("No valid market data frames available for backtest.")

    settings = load_backtest_settings()
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(settings["initial_cash"])
    cerebro.broker.setcommission(commission=settings["commission"])
    cerebro.broker.set_slippage_perc(settings["slippage_perc"])

    for symbol, frame in frames.items():
        data_feed = SignalPandasData(dataname=frame, name=symbol)
        cerebro.adddata(data_feed)

    cerebro.addstrategy(
        AgentSignalStrategy,
        stop_loss_pct=settings["stop_loss_pct"],
        take_profit_pct=settings["take_profit_pct"],
        trailing_stop_pct=settings["trailing_stop_pct"],
        min_cash_pct=settings["min_cash_pct"],
        max_position_pct=settings["max_position_pct"],
    )

    initial_value = float(cerebro.broker.getvalue())
    strategies = cerebro.run()
    strategy = strategies[0]
    final_value = float(cerebro.broker.getvalue())
    trades = strategy.closed_trades
    win_count = sum(1 for item in trades if item["pnl_amount"] > 0)
    metrics = {
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return_pct": (final_value - initial_value) / initial_value if initial_value else 0.0,
        "trade_count": len(trades),
        "win_rate": win_count / len(trades) if trades else 0.0,
    }

    output_stem = signal_path.stem.replace(".pipeline", "")
    json_output = REPORT_DIR / f"{output_stem}.backtest.json"
    md_output = REPORT_DIR / f"{output_stem}.backtest.md"

    summarize_backtest_result(snapshot, metrics, trades, json_output)
    render_backtest_markdown(snapshot, metrics, trades, md_output)

    print(f"[DONE] 回测 JSON 已生成: {json_output}")
    print(f"[DONE] 回测 Markdown 已生成: {md_output}")
    print(f"[INFO] final_value={final_value:.2f} total_return_pct={metrics['total_return_pct'] * 100:.2f}%")


if __name__ == "__main__":
    main()
