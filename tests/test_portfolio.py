from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.portfolio import load_positions, positions_to_map


class PortfolioSmokeTest(unittest.TestCase):
    def test_load_positions_normalizes_types(self) -> None:
        content = """positions:
  - symbol: "2074"
    name: "国轩高科"
    cost: "39.403"
    shares: "1000"
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "positions.yaml"
            path.write_text(content, encoding="utf-8")
            positions = load_positions(path)

        self.assertEqual(positions[0]["symbol"], "002074")
        self.assertEqual(positions[0]["cost"], 39.403)
        self.assertEqual(positions[0]["shares"], 1000)

    def test_positions_to_map(self) -> None:
        positions = [{"symbol": "2074", "name": "国轩高科", "cost": 39.4, "shares": 1000}]
        mapping = positions_to_map(positions)

        self.assertIn("002074", mapping)
        self.assertEqual(mapping["002074"]["name"], "国轩高科")


if __name__ == "__main__":
    unittest.main()
