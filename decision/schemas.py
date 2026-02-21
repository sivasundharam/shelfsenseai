from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionOutput(BaseModel):
    alert: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_action: str = "No action"
    reason: str = "Insufficient confidence"
    tags: list[str] = Field(default_factory=list)


class RCIOutput(BaseModel):
    issue: str = "No issue"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_action: str = "Monitor"
    reason: str = "No cluster signal"
