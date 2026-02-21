from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class ModulateSTTClient:
    """
    Modulate Velma-2 batch STT client.

    Based on hackathon docs in `hackathon_docs/`:
    - endpoint: /api/velma-2-stt-batch
    - auth: X-API-Key header
    - multipart field: upload_file
    """

    def __init__(self, api_key: str, endpoint: str, enabled: bool) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.enabled = enabled and bool(api_key)

    def transcribe_file(self, audio_path: Path) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if not audio_path.exists() or not audio_path.is_file():
            LOGGER.warning("Modulate STT audio file missing: %s", audio_path)
            return None

        headers = {"X-API-Key": self.api_key}
        data = {
            "speaker_diarization": "true",
            # Keep these off to preserve ShelfSense privacy policy.
            "emotion_signal": "false",
            "accent_signal": "false",
            "pii_phi_tagging": "false",
        }

        for attempt in range(2):
            try:
                with audio_path.open("rb") as f:
                    files = {
                        "upload_file": (
                            audio_path.name,
                            f,
                            "application/octet-stream",
                        )
                    }
                    resp = requests.post(
                        self.endpoint,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=45,
                    )
                if resp.status_code >= 300:
                    LOGGER.warning("Modulate STT failed status=%s body=%s", resp.status_code, resp.text[:300])
                    time.sleep(0.7 * (attempt + 1))
                    continue
                payload = resp.json()
                return payload if isinstance(payload, dict) else None
            except Exception as exc:
                LOGGER.warning("Modulate STT request failed: %s", exc)
                time.sleep(0.7 * (attempt + 1))
        return None
