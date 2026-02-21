from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)


class ModulateClient:
    def __init__(self, api_key: str, endpoint: str) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.enabled = bool(api_key)

    def synthesize(self, text: str) -> Path | None:
        if not self.enabled:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=8)
            if resp.status_code >= 300:
                LOGGER.warning("Modulate failed status=%s body=%s", resp.status_code, resp.text[:200])
                return None
            suffix = ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(resp.content)
                return Path(f.name)
        except Exception as exc:
            LOGGER.warning("Modulate request failed: %s", exc)
            return None
