from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.market_data import fetch_stock_daily, load_watchlist, save_daily_to_parquet


def parse_args() -> argparse.Namespace:
    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    parser = argparse.ArgumentParser(description="Sync daily A-share market data into data/raw parquet files.")
    parser.add_argument("--start", default=default_start.isoformat(), help="Start date, e.g. 2025-01-01.")
    parser.add_argument("--end", default=default_end.isoformat(), help="End date, e.g. 2026-04-20.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of watchlist symbols to sync.")
    parser.add_argument("--symbols", nargs="*", help="Optional explicit symbols. Defaults to config/universe.yaml watchlist.")
    return parser.parse_args()


def sync_market_data(symbols: list[str] | None, start: str, end: str, limit: int) -> list[Path]:
    if symbols:
        selected = [{"symbol": symbol} for symbol in symbols]
    else:
        selected = load_watchlist()

    if limit > 0:
        selected = selected[:limit]

    if not selected:
        raise RuntimeError("No symbols to sync. Add items to config/universe.yaml watchlist or pass --symbols.")

    saved_paths: list[Path] = []
    for item in selected:
        symbol = str(item["symbol"]).zfill(6)
        name = item.get("name", "")
        label = f"{symbol} {name}".strip()

        print(f"[INFO] 开始同步 {label} from {start} to {end}")
        try:
            df = fetch_stock_daily(symbol, start, end)
        except Exception as exc:
            print(f"[WARN] {symbol} 数据同步失败: {exc}")
            continue

        if df.empty:
            print(f"[WARN] {symbol} 数据缺失")
            continue

        output_path = save_daily_to_parquet(df, symbol)
        saved_paths.append(output_path)
        print(f"[INFO] Saved {len(df)} rows to {output_path.relative_to(PROJECT_ROOT)}")

    return saved_paths


def main() -> None:
    args = parse_args()
    saved_paths = sync_market_data(args.symbols, args.start, args.end, args.limit)
    print(f"[DONE] 行情同步完成，保存 {len(saved_paths)} 个 parquet 文件")


if __name__ == "__main__":
    main()
