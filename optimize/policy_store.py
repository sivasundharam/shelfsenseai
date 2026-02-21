from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from config import append_jsonl


@dataclass(slots=True)
class Policy:
    alert_conf_threshold: float = 0.75
    dwell_threshold_sec: float = 20.0
    motion_threshold: float = 0.25
    rci_min_alerts: int = 3
    policy_version: int = 1


class PolicyStore:
    def __init__(self, policy_path: Path, changes_path: Path) -> None:
        self.policy_path = policy_path
        self.changes_path = changes_path
        if not self.policy_path.exists():
            self.save(Policy())

    def load(self) -> Policy:
        try:
            raw = json.loads(self.policy_path.read_text(encoding="utf-8"))
            return Policy(**raw)
        except Exception:
            return Policy()

    def save(self, policy: Policy, reason: str = "init") -> None:
        self.policy_path.write_text(json.dumps(asdict(policy), ensure_ascii=True, indent=2), encoding="utf-8")
        append_jsonl(
            self.changes_path,
            {
                "ts": time.time(),
                "reason": reason,
                "policy": asdict(policy),
            },
        )
