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


def _normalize_symbol(value: Any) -> str:
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    return digits.zfill(6) if digits else str(value).strip().zfill(6)


def _normalize_position(position: dict[str, Any]) -> dict[str, Any]:
    if not position.get("symbol"):
        raise ValueError(f"invalid position item: {position!r}")

    normalized = dict(position)
    normalized["symbol"] = _normalize_symbol(position["symbol"])
    normalized["name"] = str(position.get("name") or normalized["symbol"])

    cost = pd.to_numeric(pd.Series([position.get("cost")]), errors="coerce").iloc[0]
    shares = pd.to_numeric(pd.Series([position.get("shares")]), errors="coerce").iloc[0]
    normalized["cost"] = float(cost) if not pd.isna(cost) else 0.0
    normalized["shares"] = int(shares) if not pd.isna(shares) else 0
    return normalized


def load_positions(path: Path) -> list[dict[str, Any]]:
    cfg = load_yaml(path)
    positions = cfg.get("positions", [])
    if not isinstance(positions, list):
        raise ValueError("positions must be a list")
    return [_normalize_position(position) for position in positions]


def positions_to_map(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position in positions:
        if not isinstance(position, dict) or not position.get("symbol"):
            raise ValueError(f"invalid position item: {position!r}")
        symbol = _normalize_symbol(position["symbol"])
        result[symbol] = {**position, "symbol": symbol}
    return result
