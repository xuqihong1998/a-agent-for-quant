from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.data_loader import load_symbol_df, load_watchlist
from tools.indicators import calc_basic_indicators
from tools.portfolio import load_positions, positions_to_map
from tools.report_writer import enrich_position_info, summarize_symbol
from tools.risk_checks import generate_symbol_plan


CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "daily"
PLAN_ACTIONS = ["继续持有", "谨慎持有", "重点观察", "普通观察", "减仓观察", "暂不关注"]


def render_plan_markdown(plans: list[dict[str, Any]], output_path: Path) -> None:
    today = output_path.stem.removesuffix("-plan") or datetime.now().strftime("%Y-%m-%d")
    lines = [f"# 次日交易计划（{today}）\n"]

    groups: dict[str, list[dict[str, Any]]] = {action: [] for action in PLAN_ACTIONS}
    for plan in plans:
        groups.setdefault(plan["action"], []).append(plan)

    for action, items in groups.items():
        lines.append(f"## {action}")
        if not items:
            lines.append("- 暂无\n")
            continue

        for plan in items:
            title = plan["symbol"] if plan["name"] == plan["symbol"] else f"{plan['symbol']} {plan['name']}"
            lines.append(
                f"""### {title}
- 关注价位：{plan['focus_price']}
- 入场/处理条件：{plan['entry_condition']}
- 失效条件：{plan['invalid_condition']}
- 风险备注：{plan['risk_note']}
"""
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    watchlist = load_watchlist(CONFIG_DIR)
    positions = load_positions(CONFIG_DIR / "positions.yaml")
    position_map = positions_to_map(positions)

    plans: list[dict[str, Any]] = []
    for item in watchlist:
        symbol = item["symbol"]
        name = item["name"]

        try:
            print(f"[INFO] 开始处理 {symbol}")
            df = load_symbol_df(DATA_DIR, symbol)
            df = calc_basic_indicators(df)
            summary = summarize_symbol(symbol, df, name=name)

            if symbol in position_map:
                summary = enrich_position_info(summary, position_map[symbol])
            else:
                summary["is_position"] = False

            plans.append(generate_symbol_plan(summary))
        except Exception as exc:
            print(f"[WARN] {symbol} 生成计划失败: {exc}")

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = REPORT_DIR / f"{today}-plan.md"
    render_plan_markdown(plans, output_path)
    print(f"[DONE] 次日计划生成完成: {output_path}")


if __name__ == "__main__":
    main()
