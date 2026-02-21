from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class ZoneVisit:
    zone: str
    entry_ts: float
    exit_ts: float | None = None


@dataclass(slots=True)
class PersonState:
    person_id: int
    current_zone: str
    zone_entry_ts: float
    dwell_time_sec: float
    motion_score: float
    last_seen_ts: float
    last_center: tuple[float, float]
    alert_sent_for_zone: dict[str, bool] = field(default_factory=dict)
    zone_history: list[ZoneVisit] = field(default_factory=list)


class PersonStateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._people: dict[int, PersonState] = {}

    def update_person(
        self,
        person_id: int,
        zone: str,
        center: tuple[float, float],
        ts: float,
    ) -> PersonState:
        with self._lock:
            ps = self._people.get(person_id)
            if ps is None:
                ps = PersonState(
                    person_id=person_id,
                    current_zone=zone,
                    zone_entry_ts=ts,
                    dwell_time_sec=0.0,
                    motion_score=0.0,
                    last_seen_ts=ts,
                    last_center=center,
                    alert_sent_for_zone={zone: False},
                    zone_history=[ZoneVisit(zone=zone, entry_ts=ts)],
                )
                self._people[person_id] = ps
                return ps

            disp = ((center[0] - ps.last_center[0]) ** 2 + (center[1] - ps.last_center[1]) ** 2) ** 0.5
            # Keep motion score unitless and stable across FPS; using velocity (disp/dt)
            # made scores too large and prevented event triggering.
            motion = disp
            ps.motion_score = 0.7 * ps.motion_score + 0.3 * motion
            ps.last_center = center
            ps.last_seen_ts = ts

            # Ignore transient "Unknown" classifications to avoid dwell resets from
            # noisy boundary detections.
            if zone == "Unknown" and ps.current_zone != "Unknown":
                zone = ps.current_zone

            if zone != ps.current_zone:
                if ps.zone_history:
                    ps.zone_history[-1].exit_ts = ts
                ps.current_zone = zone
                ps.zone_entry_ts = ts
                ps.alert_sent_for_zone.setdefault(zone, False)
                ps.zone_history.append(ZoneVisit(zone=zone, entry_ts=ts))

            ps.dwell_time_sec = ts - ps.zone_entry_ts
            return ps

    def prune_stale(self, timeout_sec: float, now_ts: float | None = None) -> list[int]:
        now_ts = now_ts or time.time()
        stale_ids: list[int] = []
        with self._lock:
            for pid, ps in list(self._people.items()):
                if now_ts - ps.last_seen_ts > timeout_sec:
                    stale_ids.append(pid)
                    del self._people[pid]
        return stale_ids

    def mark_alert_sent(self, person_id: int, zone: str) -> None:
        with self._lock:
            ps = self._people.get(person_id)
            if ps is not None:
                ps.alert_sent_for_zone[zone] = True

    def active_people(self) -> list[PersonState]:
        with self._lock:
            return [p for p in self._people.values()]
