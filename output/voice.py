from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from output.modulate_client import ModulateClient

LOGGER = logging.getLogger(__name__)


class VoiceNotifier:
    def __init__(self, client: ModulateClient, enabled: bool) -> None:
        self._client = client
        self._enabled = enabled

    def speak(self, text: str) -> None:
        if not self._enabled:
            return
        audio = self._client.synthesize(text)
        if audio is None:
            LOGGER.info("Voice fallback text: %s", text)
            return
        self._play(audio)

    def _play(self, audio_path: Path) -> None:
        for cmd in (["afplay", str(audio_path)], ["ffplay", "-nodisp", "-autoexit", str(audio_path)]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
        LOGGER.info("Saved voice output at %s", audio_path)
