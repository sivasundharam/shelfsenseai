from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    braintrust_api_key: str = os.getenv("BRAINTRUST_API_KEY", "")
    braintrust_project: str = os.getenv("BRAINTRUST_PROJECT", "")
    modulate_api_key: str = os.getenv("MODULATE_API_KEY", "")
    modulate_stt_url: str = os.getenv("MODULATE_STT_URL", "https://modulate-prototype-apis.com/api/velma-2-stt-batch")
    modulate_stt_audio_path: str = os.getenv("MODULATE_STT_AUDIO_PATH", "")
    modulate_stt_enabled: bool = os.getenv("MODULATE_STT_ENABLED", "false").lower() == "true"
    video_source: str = os.getenv("VIDEO_SOURCE", "webcam")
    run_fps: int = int(os.getenv("RUN_FPS", "10"))
    event_check_interval_sec: float = float(os.getenv("EVENT_CHECK_INTERVAL_SEC", "1.0"))
    rci_window_sec: int = int(os.getenv("RCI_WINDOW_SEC", "300"))
    rci_min_alerts: int = int(os.getenv("RCI_MIN_ALERTS", "3"))
    alert_conf_threshold: float = float(os.getenv("ALERT_CONF_THRESHOLD", "0.75"))
    dwell_threshold_sec: float = float(os.getenv("DWELL_THRESHOLD_SEC", "20"))
    motion_threshold: float = float(os.getenv("MOTION_THRESHOLD", "0.25"))
    track_lost_timeout_sec: float = float(os.getenv("TRACK_LOST_TIMEOUT_SEC", "1.0"))
    voice_enabled: bool = os.getenv("VOICE_ENABLED", "false").lower() == "true"
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    modulate_tts_url: str = os.getenv("MODULATE_TTS_URL", "https://api.modulate.ai/v1/tts")
    simulation_mode: bool = os.getenv("SIMULATION_MODE", "false").lower() == "true"
    simulation_events_path: str = os.getenv("SIMULATION_EVENTS_PATH", "demo/sim_events.json")
    show_debug_window: bool = os.getenv("SHOW_DEBUG_WINDOW", "true").lower() == "true"
    policy_override_from_env: bool = os.getenv("POLICY_OVERRIDE_FROM_ENV", "false").lower() == "true"
    outcome_eval_delay_sec: float = float(os.getenv("OUTCOME_EVAL_DELAY_SEC", "45"))
    optimize_every_n_events: int = int(os.getenv("OPTIMIZE_EVERY_N_EVENTS", "20"))
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    rci_cooldown_sec: int = 300
    runtime_dir: Path = field(default_factory=lambda: Path("runtime"))

    # Rectangles are in normalized [x1, y1, x2, y2]
    zones: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "Aisle_1": (0.05, 0.12, 0.45, 0.85),
            "Aisle_2": (0.46, 0.12, 0.82, 0.85),
            "Checkout": (0.83, 0.12, 0.98, 0.85),
        }
    )


SETTINGS = Settings()


def ensure_runtime_files() -> None:
    SETTINGS.runtime_dir.mkdir(parents=True, exist_ok=True)
    for rel, default in [
        ("alerts.jsonl", ""),
        ("rci.jsonl", ""),
        ("events.jsonl", ""),
        ("outcomes.jsonl", ""),
        ("staff_feedback.jsonl", ""),
        ("policy_changes.jsonl", ""),
        ("braintrust_log.jsonl", ""),
        ("metrics.json", "{}"),
        ("state.json", "{}"),
    ]:
        p = SETTINGS.runtime_dir / rel
        if not p.exists():
            p.write_text(default, encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except (json.JSONDecodeError, OSError):
        return default


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


ensure_runtime_files()
