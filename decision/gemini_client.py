from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.enabled = bool(api_key)
        self._fallback_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-flash-latest"]

    def generate_json(self, system_prompt: str, user_payload: dict[str, Any], timeout_sec: float = 12.0) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        prompt = (
            f"System: {system_prompt}\n"
            "Return JSON only with no markdown fence.\n"
            f"Observation: {json.dumps(user_payload, ensure_ascii=True)}"
        )

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"},
        }

        model_candidates = [self.model] + [m for m in self._fallback_models if m != self.model]
        model_index = 0
        for attempt in range(4):
            model_name = model_candidates[min(model_index, len(model_candidates) - 1)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(url, json=body, timeout=timeout_sec)
                if resp.status_code >= 400:
                    LOGGER.warning("Gemini HTTP error %s: %s", resp.status_code, resp.text[:300])
                    if resp.status_code == 404 and "models/" in resp.text and model_index + 1 < len(model_candidates):
                        model_index += 1
                        self.model = model_candidates[model_index]
                        LOGGER.warning("Switching Gemini model fallback to %s", self.model)
                    time.sleep(0.7 * (attempt + 1))
                    continue
                payload = resp.json()
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except Exception as exc:
                LOGGER.warning("Gemini call failed attempt=%d reason=%s", attempt + 1, exc)
                time.sleep(0.7 * (attempt + 1))
        return None
