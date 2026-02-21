from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from api.models import (
    AudioFeedbackResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    PolicyResponse,
    PolicyUpdateRequest,
    StateResponse,
)
from config import SETTINGS
from optimize.policy_store import Policy, PolicyStore
from output.modulate_stt_client import ModulateSTTClient

router = APIRouter()


RUNTIME = Path("runtime")
_STT = ModulateSTTClient(api_key=SETTINGS.modulate_api_key, endpoint=SETTINGS.modulate_stt_url, enabled=SETTINGS.modulate_stt_enabled)
_POLICY_STORE = PolicyStore(policy_path=RUNTIME / "policy.json", changes_path=RUNTIME / "policy_changes.jsonl")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return default


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[dict] = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _map_feedback_from_text(transcript: str) -> str:
    text = transcript.lower()
    false_markers = [
        "false alert",
        "wrong alert",
        "not needed",
        "unnecessary",
        "no help needed",
    ]
    appreciate_markers = [
        "appreciate",
        "good catch",
        "great alert",
        "helpful",
    ]
    thanks_markers = [
        "thank you",
        "thanks",
    ]
    if any(m in text for m in false_markers):
        return "false_alert"
    if any(m in text for m in appreciate_markers):
        return "appreciate_it"
    if any(m in text for m in thanks_markers):
        return "thanks"
    return "no_response"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    policy = _read_json(RUNTIME / "policy.json", {})
    return StateResponse(policy=policy, last_alerts=_tail_jsonl(RUNTIME / "alerts.jsonl", 20), last_rci=_tail_jsonl(RUNTIME / "rci.jsonl", 10))


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    data = _read_json(RUNTIME / "metrics.json", {})
    return MetricsResponse(metrics=data)


@router.get("/feedback", response_model=FeedbackListResponse)
def feedback(limit: int = 50) -> FeedbackListResponse:
    n = max(1, min(limit, 500))
    items = _tail_jsonl(RUNTIME / "staff_feedback.jsonl", n)
    return FeedbackListResponse(items=items)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    alert_id = body.alert_id.strip()
    note = body.note.strip()
    item = {
        "ts": time.time(),
        "alert_id": alert_id,
        "feedback": body.feedback,
        "note": note,
    }
    _append_jsonl(RUNTIME / "staff_feedback.jsonl", item)
    return FeedbackResponse(status="ok", item=item)


@router.post("/feedback/audio", response_model=AudioFeedbackResponse)
async def submit_audio_feedback(
    alert_id: str = Form(...),
    audio: UploadFile = File(...),
) -> AudioFeedbackResponse:
    suffix = Path(audio.filename or "feedback_audio.wav").suffix or ".wav"
    transcript = ""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await audio.read()
            tmp.write(data)
            tmp_path = Path(tmp.name)
        payload = _STT.transcribe_file(tmp_path)
        if isinstance(payload, dict):
            transcript = str(payload.get("text", "")).strip()
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    mapped = _map_feedback_from_text(transcript)
    item = {
        "ts": time.time(),
        "alert_id": alert_id.strip(),
        "feedback": mapped,
        "note": "",
        "source": "audio",
        "transcript": transcript,
    }
    _append_jsonl(RUNTIME / "staff_feedback.jsonl", item)
    return AudioFeedbackResponse(status="ok", item=item, transcript=transcript)


@router.post("/policy/update", response_model=PolicyResponse)
def update_policy(body: PolicyUpdateRequest) -> PolicyResponse:
    policy = _POLICY_STORE.load()
    changed = False

    if body.alert_conf_threshold is not None and body.alert_conf_threshold != policy.alert_conf_threshold:
        policy.alert_conf_threshold = float(body.alert_conf_threshold)
        changed = True
    if body.dwell_threshold_sec is not None and body.dwell_threshold_sec != policy.dwell_threshold_sec:
        policy.dwell_threshold_sec = float(body.dwell_threshold_sec)
        changed = True
    if body.motion_threshold is not None and body.motion_threshold != policy.motion_threshold:
        policy.motion_threshold = float(body.motion_threshold)
        changed = True
    if body.rci_min_alerts is not None and body.rci_min_alerts != policy.rci_min_alerts:
        policy.rci_min_alerts = int(body.rci_min_alerts)
        changed = True

    if changed:
        policy.policy_version += 1
        _POLICY_STORE.save(policy, reason="manual_ui_update")
        return PolicyResponse(status="ok", policy=asdict(policy))
    return PolicyResponse(status="no_change", policy=asdict(policy))


@router.post("/policy/reset", response_model=PolicyResponse)
def reset_policy() -> PolicyResponse:
    current = _POLICY_STORE.load()
    baseline = Policy(policy_version=max(1, current.policy_version + 1))
    _POLICY_STORE.save(baseline, reason="manual_ui_reset")
    return PolicyResponse(status="ok", policy=asdict(baseline))
