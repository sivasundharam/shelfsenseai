from __future__ import annotations

import json
import os
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
POLICY_CHANGES_PATH = Path("runtime/policy_changes.jsonl")


def fetch(path: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=2)
        if resp.status_code < 300:
            return resp.json()
    except Exception:
        pass
    return {}


def post(path: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=3)
        if resp.status_code < 300:
            return resp.json()
    except Exception:
        pass
    return {}


def post_audio_feedback(alert_id: str, audio_bytes: bytes, filename: str, mime_type: str) -> dict:
    try:
        files = {"audio": (filename, BytesIO(audio_bytes), mime_type or "application/octet-stream")}
        data = {"alert_id": alert_id}
        resp = requests.post(f"{API_BASE}/feedback/audio", files=files, data=data, timeout=40)
        if resp.status_code < 300:
            return resp.json()
    except Exception:
        pass
    return {}


def read_policy_changes(limit: int = 50) -> list[dict]:
    if not POLICY_CHANGES_PATH.exists():
        return []
    lines = POLICY_CHANGES_PATH.read_text(encoding="utf-8").strip().splitlines()
    rows: list[dict] = []
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def _format_alert(alert: dict) -> str:
    return (
        f"[{alert.get('zone', 'Unknown')}] conf={alert.get('confidence', 0):.2f} "
        f"action={alert.get('action', 'N/A')}"
    )


st.set_page_config(page_title="ShelfSense AI Dashboard", layout="wide")
st.title("ShelfSense AI - Live Dashboard")

with st.sidebar:
    st.subheader("Controls")
    auto_refresh = st.toggle("Auto refresh", value=True)
    refresh_sec = st.slider("Refresh interval (sec)", 1, 15, 3)
    manual = st.button("Refresh now")

state = fetch("/state")
metrics_payload = fetch("/metrics")
metrics = metrics_payload.get("metrics", {}) if isinstance(metrics_payload, dict) else {}
policy_changes = read_policy_changes(limit=80)

alerts = state.get("last_alerts", []) if isinstance(state, dict) else []
rci = state.get("last_rci", []) if isinstance(state, dict) else []
policy = state.get("policy", {}) if isinstance(state, dict) else {}
feedback_data = fetch("/feedback?limit=100")
feedback_items = feedback_data.get("items", []) if isinstance(feedback_data, dict) else []

zones = sorted({a.get("zone", "Unknown") for a in alerts if isinstance(a, dict)})
zone_filter = st.multiselect("Filter zones", options=zones, default=zones)
filtered_alerts = [a for a in alerts if a.get("zone") in zone_filter] if zone_filter else alerts

latest_alert_id = filtered_alerts[-1].get("alert_id") if filtered_alerts else None
last_seen = st.session_state.get("last_seen_alert_id")
if latest_alert_id and latest_alert_id != last_seen:
    st.session_state["last_seen_alert_id"] = latest_alert_id
    newest = filtered_alerts[-1]
    st.error(f"New Alert: {_format_alert(newest)}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts (view)", len(filtered_alerts))
c2.metric("RCI recs", len(rci))
c3.metric("Avg score (50)", f"{metrics.get('avg_score_last_50', 0.0):.3f}")
c4.metric("Resolved rate", f"{metrics.get('resolved_rate', 0.0):.2%}")

left, right = st.columns([2, 1])

with left:
    st.subheader("Live Alerts")
    st.dataframe(filtered_alerts, use_container_width=True, hide_index=True)

    st.subheader("Staff Feedback")
    if filtered_alerts:
        alert_options: dict[str, dict] = {}
        for alert in reversed(filtered_alerts):
            alert_id = str(alert.get("alert_id", ""))
            if not alert_id:
                continue
            label = (
                f"{alert_id[:8]} | {alert.get('zone', 'Unknown')} | "
                f"conf={float(alert.get('confidence', 0.0)):.2f} | {alert.get('action', '')}"
            )
            alert_options[label] = alert

        if alert_options:
            selected_label = st.selectbox("Select alert", options=list(alert_options.keys()), key="feedback_alert_select")
            selected_alert = alert_options[selected_label]
            selected_alert_id = str(selected_alert.get("alert_id", ""))
            st.caption("Record staff voice feedback and submit. The backend transcribes + maps to feedback label.")
            audio_capture = st.audio_input("Record staff feedback", key="staff_audio_input")
            uploaded_audio = st.file_uploader(
                "Or upload audio (wav/mp3/m4a)",
                type=["wav", "mp3", "m4a", "aac", "ogg"],
                key="staff_audio_upload",
            )
            if st.button("Submit audio feedback", use_container_width=True):
                file_obj = audio_capture if audio_capture is not None else uploaded_audio
                if file_obj is None:
                    st.warning("Please record or upload audio first.")
                else:
                    payload = post_audio_feedback(
                        alert_id=selected_alert_id,
                        audio_bytes=file_obj.getvalue(),
                        filename=getattr(file_obj, "name", "staff_feedback.wav"),
                        mime_type=getattr(file_obj, "type", "application/octet-stream"),
                    )
                    if payload.get("status") == "ok":
                        item = payload.get("item", {})
                        st.success(f"Saved audio feedback: {item.get('feedback', 'no_response')}")
                        transcript = str(payload.get("transcript", "")).strip()
                        if transcript:
                            st.caption(f'Transcript: "{transcript}"')
                        st.rerun()
                    else:
                        st.error("Audio feedback submit failed. Check API logs and Modulate STT settings.")
        else:
            st.info("Alerts are missing `alert_id`, cannot attach feedback.")
    else:
        st.info("No alerts yet.")

    if feedback_items:
        st.dataframe(list(reversed(feedback_items[-30:])), use_container_width=True, hide_index=True)
    else:
        st.caption("No staff feedback recorded yet.")

    st.subheader("Root Cause Recommendations")
    st.dataframe(rci, use_container_width=True, hide_index=True)

with right:
    st.subheader("Demo Threshold Controls")
    conf_default = float(policy.get("alert_conf_threshold", 0.75))
    dwell_default = float(policy.get("dwell_threshold_sec", 20.0))
    motion_default = float(policy.get("motion_threshold", 0.25))
    rci_default = int(policy.get("rci_min_alerts", 3))

    demo_conf = st.slider("Alert confidence threshold", min_value=0.50, max_value=0.98, value=min(max(conf_default, 0.50), 0.98), step=0.01)
    demo_dwell = st.slider("Dwell threshold (sec)", min_value=3.0, max_value=60.0, value=min(max(dwell_default, 3.0), 60.0), step=1.0)
    demo_motion = st.slider("Motion threshold", min_value=0.05, max_value=1.0, value=min(max(motion_default, 0.05), 1.0), step=0.01)
    demo_rci = st.slider("RCI min alerts", min_value=1, max_value=10, value=min(max(rci_default, 1), 10), step=1)
    cols_demo = st.columns(2)
    if cols_demo[0].button("Apply thresholds", use_container_width=True):
        payload = post(
            "/policy/update",
            {
                "alert_conf_threshold": float(demo_conf),
                "dwell_threshold_sec": float(demo_dwell),
                "motion_threshold": float(demo_motion),
                "rci_min_alerts": int(demo_rci),
            },
        )
        if payload.get("status") in {"ok", "no_change"}:
            st.success(f"Policy {payload.get('status')}.")
            st.rerun()
        else:
            st.error("Failed to apply thresholds.")
    if cols_demo[1].button("Reset baseline", use_container_width=True):
        payload = post("/policy/reset", {})
        if payload.get("status") == "ok":
            st.success("Policy reset to baseline defaults.")
            st.rerun()
        else:
            st.error("Policy reset failed.")

    st.subheader("Current Policy")
    st.json(policy)

    st.subheader("Eval Metrics")
    st.json(metrics)

    st.subheader("Optimization Summary")
    if policy_changes:
        latest = policy_changes[-1]
        st.caption(f"Latest reason: {latest.get('reason', 'n/a')}")
        st.caption(f"Policy version: {policy.get('policy_version', 'n/a')}")
    else:
        st.caption("No policy changes logged yet.")

st.subheader("Alert Confidence Trend")
confidence_series = [float(a.get("confidence", 0.0)) for a in filtered_alerts]
if confidence_series:
    st.line_chart(confidence_series)
else:
    st.info("No alerts yet. Run pipeline or simulation to populate this chart.")

st.subheader("Alerts by Zone")
zone_counts = Counter(a.get("zone", "Unknown") for a in filtered_alerts)
if zone_counts:
    st.bar_chart({"count": zone_counts})
else:
    st.info("No zone distribution to display.")

st.subheader("Optimization History")
if policy_changes:
    compact_rows = []
    version_series: list[float] = []
    conf_series: list[float] = []
    dwell_series: list[float] = []
    for row in policy_changes[-25:]:
        pol = row.get("policy", {})
        compact_rows.append(
            {
                "ts": round(float(row.get("ts", 0.0)), 2),
                "reason": row.get("reason", ""),
                "policy_version": pol.get("policy_version"),
                "alert_conf_threshold": pol.get("alert_conf_threshold"),
                "dwell_threshold_sec": pol.get("dwell_threshold_sec"),
                "motion_threshold": pol.get("motion_threshold"),
            }
        )
        version_series.append(float(pol.get("policy_version", 0)))
        conf_series.append(float(pol.get("alert_conf_threshold", 0.0)))
        dwell_series.append(float(pol.get("dwell_threshold_sec", 0.0)))

    st.dataframe(compact_rows, use_container_width=True, hide_index=True)
    st.line_chart(
        {
            "policy_version": version_series,
            "alert_conf_threshold": conf_series,
            "dwell_threshold_sec": dwell_series,
        }
    )
else:
    st.info("Run simulation/video multiple cycles to populate optimization history.")

if auto_refresh and not manual:
    time.sleep(refresh_sec)
    st.rerun()
