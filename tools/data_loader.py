from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_watchlist(config_dir: Path) -> list[dict[str, str]]:
    cfg = load_yaml(config_dir / "universe.yaml")
    watchlist = cfg.get("watchlist", [])
    if not isinstance(watchlist, list):
        raise ValueError("watchlist must be a list")

    symbols: list[dict[str, str]] = []
    for item in watchlist:
        if isinstance(item, str):
            symbol = item.zfill(6)
            symbols.append({"symbol": symbol, "name": symbol})
            continue

        if not isinstance(item, dict) or not item.get("symbol"):
            raise ValueError(f"invalid watchlist item: {item!r}")

        symbol = str(item["symbol"]).zfill(6)
        name = str(item.get("name") or symbol)
        symbols.append({"symbol": symbol, "name": name})

    return symbols


def load_symbol_df(data_dir: Path, symbol: str) -> pd.DataFrame:
    path = data_dir / f"{str(symbol).zfill(6)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{symbol} 数据文件不存在: {path}")
    return pd.read_parquet(path)
