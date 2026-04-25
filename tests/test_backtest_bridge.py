from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from tools.backtest import (
    SignalSnapshot,
    build_signal_feed_frames,
    build_trade_plan_from_snapshot,
    load_signal_snapshot,
    render_backtest_markdown,
    signal_to_trade_plan,
    summarize_backtest_result,
)


class BacktestBridgeTest(unittest.TestCase):
    def test_signal_to_trade_plan(self) -> None:
        plan = signal_to_trade_plan(
            {
                "symbol": "2463",
                "name": "沪电股份",
                "final_action": "继续持有",
                "risk_level": "low",
                "position_pct_limit": 0.2,
                "confidence": 0.8,
                "thesis": "test",
                "plan_steps": ["a"],
                "risk_flags": ["b"],
            }
        )

        self.assertEqual(plan["symbol"], "002463")
        self.assertEqual(plan["target_pct"], 0.2)
        self.assertEqual(plan["action"], "继续持有")

    def test_load_snapshot_and_build_trade_plan(self) -> None:
        payload = {
            "plan_agent": {
                "review_date": "2026-04-25",
                "symbols": [
                    {
                        "symbol": "002463",
                        "name": "沪电股份",
                        "final_action": "继续持有",
                        "risk_level": "low",
                        "position_pct_limit": 0.2,
                        "confidence": 0.8,
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "2026-04-25.pipeline.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            snapshot = load_signal_snapshot(path)

        trade_plan = build_trade_plan_from_snapshot(snapshot)
        self.assertEqual(snapshot.review_date, "2026-04-25")
        self.assertIn("002463", trade_plan)

    def test_build_signal_feed_frames_and_render_outputs(self) -> None:
        trade_plan = {
            "002463": {
                "symbol": "002463",
                "name": "沪电股份",
                "action": "继续持有",
                "risk_level": "low",
                "target_pct": 0.2,
                "confidence": 0.8,
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "date": ["2026-04-23", "2026-04-24", "2026-04-25"],
                    "symbol": ["002463", "002463", "002463"],
                    "open": [100, 101, 102],
                    "high": [101, 102, 103],
                    "low": [99, 100, 101],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1000, 1100, 1200],
                    "amount": [100000, 110000, 120000],
                }
            )
            df.to_parquet(data_dir / "002463.parquet", index=False)
            frames = build_signal_feed_frames(trade_plan, data_dir=data_dir)
            self.assertIn("002463", frames)
            self.assertIn("signal_target_pct", frames["002463"].columns)

            snapshot = SignalSnapshot(review_date="2026-04-25", symbols=[], source_path=Path("signal.json"))
            json_output = Path(tmp_dir) / "report.json"
            md_output = Path(tmp_dir) / "report.md"
            metrics = {"final_value": 1100000.0, "total_return_pct": 0.1, "trade_count": 1, "win_rate": 1.0}
            trades = [
                {
                    "symbol": "002463",
                    "name": "沪电股份",
                    "opened_at": "2026-04-24",
                    "closed_at": "2026-04-25",
                    "pnl_pct": 0.05,
                    "pnl_amount": 5000.0,
                    "exit_reason": "take_profit",
                }
            ]

            summarize_backtest_result(snapshot, metrics, trades, json_output)
            render_backtest_markdown(snapshot, metrics, trades, md_output)

            self.assertTrue(json_output.exists())
            self.assertTrue(md_output.exists())


if __name__ == "__main__":
    unittest.main()
