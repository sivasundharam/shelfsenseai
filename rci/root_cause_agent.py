from __future__ import annotations

import logging
from typing import Any

from decision.gemini_client import GeminiClient
from decision.prompts import RCI_SYSTEM_PROMPT
from decision.schemas import RCIOutput

LOGGER = logging.getLogger(__name__)


class RootCauseAgent:
    def __init__(self, gemini: GeminiClient) -> None:
        self._gemini = gemini

    def analyze(self, cluster_payload: dict[str, Any]) -> RCIOutput:
        result = self._gemini.generate_json(RCI_SYSTEM_PROMPT, cluster_payload)
        if result is None:
            return RCIOutput()
        try:
            return RCIOutput.model_validate(result)
        except Exception as exc:
            LOGGER.warning("RCI output validation failed: %s", exc)
            return RCIOutput()
