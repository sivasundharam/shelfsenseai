from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EvalSnapshot:
    total: int = 0
    avg_score_last_50: float = 0.0
    spam_rate: float = 0.0
    resolved_rate: float = 0.0


class MetricsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
