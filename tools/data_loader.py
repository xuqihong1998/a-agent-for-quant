from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tools.market_data import load_symbol_df as _load_symbol_df


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_symbol(value: Any) -> str:
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    return digits.zfill(6) if digits else str(value).strip().zfill(6)


def load_watchlist(config_dir: Path | None = None) -> list[dict[str, Any]]:
    target_dir = config_dir or DEFAULT_CONFIG_DIR
    cfg = load_yaml(target_dir / "universe.yaml")
    watchlist = cfg.get("watchlist", [])
    if not isinstance(watchlist, list):
        raise ValueError("watchlist must be a list")

    normalized: list[dict[str, Any]] = []
    for item in watchlist:
        if isinstance(item, str):
            symbol = _normalize_symbol(item)
            normalized.append({"symbol": symbol, "name": symbol})
            continue

        if not isinstance(item, dict) or not item.get("symbol"):
            raise ValueError(f"invalid watchlist item: {item!r}")

        symbol = _normalize_symbol(item["symbol"])
        normalized.append({**item, "symbol": symbol, "name": str(item.get("name") or symbol)})

    return normalized


def load_symbol_df(data_dir: Path, symbol: str) -> pd.DataFrame:
    return _load_symbol_df(symbol, data_dir=data_dir)
