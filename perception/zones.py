from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ZoneRect:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class ZoneMapper:
    def __init__(self, zone_defs: dict[str, tuple[float, float, float, float]]) -> None:
        self._zones = [ZoneRect(name=k, x1=v[0], y1=v[1], x2=v[2], y2=v[3]) for k, v in zone_defs.items()]

    def point_to_zone(self, x: float, y: float) -> str:
        for zone in self._zones:
            if zone.contains(x, y):
                return zone.name
        return "Unknown"

    def bbox_to_zone(self, bbox: tuple[int, int, int, int], width: int, height: int) -> str:
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2.0) / max(width, 1)
        cy = ((y1 + y2) / 2.0) / max(height, 1)
        return self.point_to_zone(cx, cy)

    def to_pixel_rects(self, width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
        out: list[tuple[str, tuple[int, int, int, int]]] = []
        for z in self._zones:
            out.append(
                (
                    z.name,
                    (
                        int(z.x1 * width),
                        int(z.y1 * height),
                        int(z.x2 * width),
                        int(z.y2 * height),
                    ),
                )
            )
        return out
