"""Score-breakdown helpers.

The MATLAB project had year-specific classes. The Python parity layer keeps a
raw-data-first representation and exposes common helpers used by LatentStrat and
the analysis tests. Unknown fields are intentionally ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def camel_to_snake(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


@dataclass
class BreakdownBase:
    raw_data: dict[str, Any]
    alliance_color: str

    def get(self, field_name: str, default: Any = 0) -> Any:
        if field_name in self.raw_data:
            return self.raw_data[field_name]
        snake_name = camel_to_snake(field_name)
        if snake_name in self.raw_data:
            return self.raw_data[snake_name]
        return default

    def require(self, field_name: str) -> Any:
        value = self.get(field_name, None)
        if value is None:
            raise KeyError(f"Missing breakdown field {field_name!r} for {self.alliance_color}")
        return value

    def get_auto_score(self) -> float:
        return float(self.get("totalAutoPoints", self.get("autoPoints", 0)) or 0)

    def get_teleop_score(self) -> float:
        return float(self.get("totalTeleopPoints", self.get("teleopPoints", 0)) or 0)

    def get_total_score(self) -> float:
        return float(self.get("totalPoints", 0) or 0)

    def get_rp1(self) -> bool:
        return bool(self.get("rp1", False))

    def get_rp2(self) -> bool:
        return bool(self.get("rp2", False))

    def get_analysis_vector(self) -> tuple[list[float], list[str]]:
        labels = ["auto_score", "teleop_score", "total_score"]
        values = [self.get_auto_score(), self.get_teleop_score(), self.get_total_score()]
        return values, labels


class BreakdownFactory:
    @staticmethod
    def create(year: int | str, data: dict[str, Any] | None, color: str, *args: Any) -> BreakdownBase:
        _ = year, args
        return BreakdownBase(data or {}, color)
