from __future__ import annotations

import logging
from typing import Any

from decision.gemini_client import GeminiClient
from decision.prompts import DECISION_SYSTEM_PROMPT
from decision.schemas import DecisionOutput

LOGGER = logging.getLogger(__name__)


class DecisionAgent:
    def __init__(self, gemini: GeminiClient) -> None:
        self._gemini = gemini

    def _heuristic_decision(self, observation: dict[str, Any]) -> DecisionOutput:
        dwell = float(observation.get("dwell_time", 0.0))
        motion = float(observation.get("motion_score", 1.0))
        zone = str(observation.get("zone", "Unknown"))

        # Rule fallback for demo reliability when LLM is unavailable/invalid.
        if dwell >= 6.0 and motion <= 0.25:
            confidence = min(0.95, max(0.78, 0.72 + (dwell / 90.0) + (0.2 - min(motion, 0.2))))
            return DecisionOutput(
                alert=True,
                confidence=round(confidence, 3),
                recommended_action=f"Dispatch associate to assist in {zone}",
                reason="Heuristic fallback: prolonged dwell with low motion indicates assistance need",
                tags=["heuristic_fallback", "assistance_candidate"],
            )
        return DecisionOutput()

    def decide(self, observation: dict[str, Any]) -> DecisionOutput:
        heuristic = self._heuristic_decision(observation)
        result = self._gemini.generate_json(DECISION_SYSTEM_PROMPT, observation)
        if result is None:
            LOGGER.warning("Gemini unavailable; using heuristic decision fallback")
            return heuristic
        try:
            out = DecisionOutput.model_validate(result)
            # Guardrail: allow strong deterministic friction signals to promote alert.
            if heuristic.alert and (not out.alert or out.confidence < 0.75):
                return heuristic
            return out
        except Exception as exc:
            LOGGER.warning("Decision output validation failed: %s", exc)
            return heuristic
