from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class StateResponse(BaseModel):
    policy: dict
    last_alerts: list[dict]
    last_rci: list[dict]


class MetricsResponse(BaseModel):
    metrics: dict


FeedbackLabel = Literal["false_alert", "appreciate_it", "thanks", "no_response"]


class FeedbackRequest(BaseModel):
    alert_id: str
    feedback: FeedbackLabel
    note: str = ""


class FeedbackResponse(BaseModel):
    status: str
    item: dict


class FeedbackListResponse(BaseModel):
    items: list[dict]


class AudioFeedbackResponse(BaseModel):
    status: str
    item: dict
    transcript: str


class PolicyUpdateRequest(BaseModel):
    alert_conf_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    dwell_threshold_sec: float | None = Field(default=None, ge=1.0, le=120.0)
    motion_threshold: float | None = Field(default=None, ge=0.0, le=2.0)
    rci_min_alerts: int | None = Field(default=None, ge=1, le=20)


class PolicyResponse(BaseModel):
    status: str
    policy: dict
