from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_positions(path: Path) -> list[dict[str, Any]]:
    cfg = load_yaml(path)
    positions = cfg.get("positions", [])
    if not isinstance(positions, list):
        raise ValueError("positions must be a list")
    return positions


def positions_to_map(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position in positions:
        if not isinstance(position, dict) or not position.get("symbol"):
            raise ValueError(f"invalid position item: {position!r}")

        symbol = str(position["symbol"]).zfill(6)
        result[symbol] = {**position, "symbol": symbol}

    return result
