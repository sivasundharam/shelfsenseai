from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from perception.state import PersonState


@dataclass(slots=True)
class AssistanceCandidateEvent:
    event_id: str
    ts: float
    person_id: int
    zone: str
    dwell_time: float
    motion_score: float
    recent_zone_history: list[str]
    queue_len: int | None = None
    pickup_proxy: bool | None = None

    def to_observation(self) -> dict:
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "person_id": self.person_id,
            "zone": self.zone,
            "dwell_time": round(self.dwell_time, 2),
            "motion_score": round(self.motion_score, 3),
            "recent_zone_history": self.recent_zone_history,
            "queue_len": self.queue_len,
            "pickup_proxy": self.pickup_proxy,
        }


class EventTrigger:
    def __init__(self, dwell_threshold_sec: float, motion_threshold: float) -> None:
        self.dwell_threshold_sec = dwell_threshold_sec
        self.motion_threshold = motion_threshold

    def maybe_trigger(self, state: PersonState) -> AssistanceCandidateEvent | None:
        if state.current_zone == "Unknown":
            return None
        already_sent = state.alert_sent_for_zone.get(state.current_zone, False)
        if already_sent:
            return None
        if state.dwell_time_sec < self.dwell_threshold_sec:
            return None
        if state.motion_score > self.motion_threshold:
            return None
        recent = [z.zone for z in state.zone_history[-4:]]
        return AssistanceCandidateEvent(
            event_id=str(uuid.uuid4()),
            ts=time.time(),
            person_id=state.person_id,
            zone=state.current_zone,
            dwell_time=state.dwell_time_sec,
            motion_score=state.motion_score,
            recent_zone_history=recent,
        )
