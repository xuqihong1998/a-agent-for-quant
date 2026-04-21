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
from tools.report_writer import enrich_position_info, render_daily_report, summarize_symbol


CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "daily"


def main() -> None:
    watchlist = load_watchlist(CONFIG_DIR)
    positions = load_positions(CONFIG_DIR / "positions.yaml")
    position_map = positions_to_map(positions)
    results: list[dict[str, Any]] = []

    for item in watchlist:
        symbol = item["symbol"]
        name = item["name"]

        try:
            print(f"[INFO] 开始处理 {symbol}")
            df = load_symbol_df(DATA_DIR, symbol)
            df = calc_basic_indicators(df)
            result = summarize_symbol(symbol, df, name=name)

            if symbol in position_map:
                result = enrich_position_info(result, position_map[symbol])
            else:
                result["is_position"] = False

            results.append(result)
        except Exception as exc:
            print(f"[WARN] {symbol} 处理失败: {exc}")

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = REPORT_DIR / f"{today}.md"
    render_daily_report(results, output_path)
    print(f"[DONE] 报告生成完成: {output_path}")


if __name__ == "__main__":
    main()
