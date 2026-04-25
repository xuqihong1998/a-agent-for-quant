from __future__ import annotations

from collections import defaultdict
from typing import Any

import backtrader as bt


class SignalPandasData(bt.feeds.PandasData):
    lines = ("signal_target_pct", "signal_confidence")
    params = (
        ("datetime", "date"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        ("signal_target_pct", "signal_target_pct"),
        ("signal_confidence", "signal_confidence"),
    )


class AgentSignalStrategy(bt.Strategy):
    params = dict(
        stop_loss_pct=0.06,
        take_profit_pct=0.15,
        trailing_stop_pct=0.08,
        min_cash_pct=0.20,
        max_position_pct=0.20,
    )

    def __init__(self) -> None:
        self.entry_info: dict[str, dict[str, float]] = {}
        self.pending_orders: dict[str, bt.Order] = {}
        self.closed_trades: list[dict[str, Any]] = []
        self.symbol_names: dict[str, str] = {}
        self.exit_reasons: dict[str, str] = {}
        self.trade_open_dates: dict[str, str] = {}

    def log(self, message: str) -> None:
        dt = self.datas[0].datetime.date(0).isoformat() if self.datas else "unknown"
        print(f"[BT {dt}] {message}")

    def start(self) -> None:
        for data in self.datas:
            self.symbol_names[data._name] = getattr(data, "_name", "unknown")
        self.log(f"strategy start datas={len(self.datas)}")

    def notify_order(self, order: bt.Order) -> None:
        data_name = order.data._name
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            self.log(
                f"order completed symbol={data_name} side={'BUY' if order.isbuy() else 'SELL'} "
                f"price={order.executed.price:.2f} size={order.executed.size}"
            )
            if order.isbuy():
                self.entry_info[data_name] = {
                    "entry_price": float(order.executed.price),
                    "highest_price": float(order.executed.price),
                }
                self.trade_open_dates[data_name] = bt.num2date(order.executed.dt).date().isoformat()
            else:
                entry = self.entry_info.pop(data_name, {"entry_price": float(order.executed.price)})
                entry_price = float(entry["entry_price"])
                exit_price = float(order.executed.price)
                size = abs(float(order.executed.size))
                pnl_amount = (exit_price - entry_price) * size
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
                self.closed_trades.append(
                    {
                        "symbol": data_name,
                        "name": self.symbol_names.get(data_name, data_name),
                        "opened_at": self.trade_open_dates.pop(data_name, ""),
                        "closed_at": bt.num2date(order.executed.dt).date().isoformat(),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": size,
                        "pnl_amount": pnl_amount,
                        "pnl_pct": pnl_pct,
                        "exit_reason": self.exit_reasons.pop(data_name, "rebalance_or_signal"),
                    }
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"order failed symbol={data_name} status={order.getstatusname()}")

        self.pending_orders.pop(data_name, None)

    def _should_exit(self, data: SignalPandasData) -> str | None:
        symbol = data._name
        position = self.getposition(data)
        if not position.size or symbol not in self.entry_info:
            return None

        close_price = float(data.close[0])
        entry_price = self.entry_info[symbol]["entry_price"]
        highest_price = max(self.entry_info[symbol]["highest_price"], close_price)
        self.entry_info[symbol]["highest_price"] = highest_price

        if close_price <= entry_price * (1 - self.p.stop_loss_pct):
            return "stop_loss"
        if close_price >= entry_price * (1 + self.p.take_profit_pct):
            return "take_profit"
        if close_price <= highest_price * (1 - self.p.trailing_stop_pct):
            return "trailing_stop"
        if float(data.signal_target_pct[0]) <= 0:
            return "signal_exit"
        return None

    def next(self) -> None:
        portfolio_value = float(self.broker.getvalue())
        cash = float(self.broker.getcash())
        min_cash = portfolio_value * self.p.min_cash_pct

        for data in self.datas:
            symbol = data._name
            if symbol in self.pending_orders:
                continue

            exit_reason = self._should_exit(data)
            position = self.getposition(data)
            if exit_reason and position.size:
                self.exit_reasons[symbol] = exit_reason
                self.pending_orders[symbol] = self.close(data=data)
                self.log(f"exit symbol={symbol} reason={exit_reason}")
                continue

            target_pct = min(float(data.signal_target_pct[0]), self.p.max_position_pct)
            if target_pct <= 0:
                continue

            if cash <= min_cash and not position.size:
                self.log(f"skip entry symbol={symbol} reason=min_cash_guard")
                continue

            self.pending_orders[symbol] = self.order_target_percent(data=data, target=target_pct)
            self.log(f"rebalance symbol={symbol} target_pct={target_pct:.2%}")

