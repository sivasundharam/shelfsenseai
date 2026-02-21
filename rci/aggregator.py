from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class AlertRecord:
    alert_id: str
    event_id: str
    ts: float
    zone: str
    confidence: float
    dwell_time: float


class RCIAggregator:
    def __init__(self, window_sec: int, min_alerts: int, cooldown_sec: int) -> None:
        self.window_sec = window_sec
        self.min_alerts = min_alerts
        self.cooldown_sec = cooldown_sec
        self._alerts: list[AlertRecord] = []
        self._cooldown_until: dict[str, float] = {}

    def add_alert(self, rec: AlertRecord) -> None:
        self._alerts.append(rec)

    def cluster_candidates(self) -> list[dict]:
        now = time.time()
        cutoff = now - self.window_sec
        self._alerts = [a for a in self._alerts if a.ts >= cutoff]

        grouped: dict[str, list[AlertRecord]] = defaultdict(list)
        for a in self._alerts:
            grouped[a.zone].append(a)

        clusters: list[dict] = []
        for zone, alerts in grouped.items():
            if len(alerts) < self.min_alerts:
                continue
            if now < self._cooldown_until.get(zone, 0.0):
                continue

            avg_conf = sum(x.confidence for x in alerts) / len(alerts)
            avg_dwell = sum(x.dwell_time for x in alerts) / len(alerts)
            queue_stats = len(alerts) / max(self.window_sec / 60.0, 1.0)

            clusters.append(
                {
                    "zone": zone,
                    "alerts_count": len(alerts),
                    "avg_confidence": round(avg_conf, 3),
                    "avg_dwell": round(avg_dwell, 2),
                    "abandon_rate_proxy": 0.0,
                    "queue_stats": round(queue_stats, 3),
                }
            )
            self._cooldown_until[zone] = now + self.cooldown_sec
        return clusters
