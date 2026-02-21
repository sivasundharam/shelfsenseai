from __future__ import annotations

import json
import importlib
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from config import SETTINGS, append_jsonl, ensure_runtime_files
from decision.decision_agent import DecisionAgent
from decision.gemini_client import GeminiClient
from optimize.metrics import MetricsStore
from optimize.optimizer import OptimizationAgent
from optimize.policy_store import PolicyStore
from output.modulate_client import ModulateClient
from output.modulate_stt_client import ModulateSTTClient
from output.voice import VoiceNotifier
from perception.events import EventTrigger
from perception.state import PersonStateStore
from perception.tracker import MultiObjectTracker
from perception.zones import ZoneMapper
from rci.aggregator import AlertRecord, RCIAggregator
from rci.root_cause_agent import RootCauseAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("shelfsense")

BraintrustClient = importlib.import_module("eval.braintrust_client").BraintrustClient


def _video_source(src: str) -> int | str:
    return 0 if src == "webcam" else src


def _stream_time_seconds(cap: cv2.VideoCapture, video_source: str) -> float:
    if video_source == "webcam":
        return time.time()
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec and pos_msec > 0:
        return pos_msec / 1000.0
    return time.time()


def _draw_overlay(
    frame,
    tracks: list[dict],
    zone_mapper: ZoneMapper,
    policy: dict,
    last_alert_text: str,
) -> None:
    h, w = frame.shape[:2]
    for name, rect in zone_mapper.to_pixel_rects(w, h):
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cv2.putText(frame, name, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 255), 1)

    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        label = f"id={tr['person_id']} z={tr['zone']} dwell={tr['dwell']:.1f}s m={tr['motion']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    pol_txt = f"conf={policy.get('alert_conf_threshold', 0.75):.2f} dwell={policy.get('dwell_threshold_sec', 20)} motion={policy.get('motion_threshold', 0.25):.2f}"
    cv2.putText(frame, pol_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 220), 2)
    cv2.putText(frame, f"last_alert={last_alert_text[:80]}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1)


def _compute_metrics(eval_records: list[dict]) -> dict[str, Any]:
    if not eval_records:
        return {
            "total": 0,
            "avg_score_last_50": 0.0,
            "spam_rate": 0.0,
            "resolved_rate": 0.0,
        }
    last50 = eval_records[-50:]
    avg_score = sum(float(r.get("scores", {}).get("overall", 0.0)) for r in last50) / len(last50)
    spam = sum(float(r.get("outcome_signals", {}).get("spam_proxy", 0.0)) for r in last50) / len(last50)
    resolved = sum(float(r.get("outcome_signals", {}).get("resolved_proxy", 0.0)) for r in last50) / len(last50)
    return {
        "total": len(eval_records),
        "avg_score_last_50": round(avg_score, 4),
        "spam_rate": round(spam, 4),
        "resolved_rate": round(resolved, 4),
    }


def _update_runtime_state(policy: dict, last_alerts: list[dict], last_rci: list[dict]) -> None:
    payload = {
        "policy": policy,
        "last_alerts": last_alerts[-20:],
        "last_rci": last_rci[-10:],
        "ts": time.time(),
    }
    (SETTINGS.runtime_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _audio_feedback_proxy(transcript_text: str) -> float:
    t = transcript_text.lower()
    false_alert_markers = [
        "false alert",
        "wrong alert",
        "not needed",
        "no help needed",
        "unnecessary alert",
    ]
    appreciation_markers = [
        "appreciate it",
        "thank you",
        "thanks",
        "helpful alert",
        "good catch",
    ]
    friction_markers = [
        "need help",
        "can't find",
        "cannot find",
        "looking for",
        "where is",
        "long line",
        "queue",
        "waiting",
    ]

    if any(k in t for k in false_alert_markers):
        return 0.0
    if any(k in t for k in appreciation_markers):
        return 1.0

    pos = sum(1 for k in friction_markers if k in t)
    score = 0.25 + 0.15 * pos
    return round(max(0.0, min(1.0, score)), 3)


def _evaluate_and_log_record(braintrust: BraintrustClient, record: dict[str, Any], sink_path: Path | None = None) -> dict[str, Any]:
    eval_record = braintrust.evaluate_record(record)
    if not eval_record.get("_braintrust_eval_success", False):
        eval_record = braintrust.fallback_score_record(eval_record)
    if sink_path is not None:
        append_jsonl(sink_path, eval_record)
    braintrust.log_record(eval_record)
    return eval_record


def _run_simulation(
    decision_agent: DecisionAgent,
    optimizer: OptimizationAgent,
    policy_store: PolicyStore,
    braintrust: BraintrustClient,
    metrics_store: MetricsStore,
) -> None:
    LOGGER.info("Running simulation mode using %s", SETTINGS.simulation_events_path)
    sim_path = Path(SETTINGS.simulation_events_path)
    if not sim_path.exists():
        LOGGER.warning("Simulation file missing: %s", sim_path)
        return
    events = json.loads(sim_path.read_text(encoding="utf-8"))
    eval_records: list[dict] = []
    zone_counts: dict[str, int] = {}
    zone_conf_proxy: dict[str, float] = {}
    for obs in events:
        policy = policy_store.load()
        out = decision_agent.decide(obs)
        alert_triggered = out.alert and out.confidence >= policy.alert_conf_threshold
        resolved_proxy = 1.0 if alert_triggered and obs.get("motion_score", 1.0) < 0.2 else 0.0
        abandoned_proxy = 1.0 if (not alert_triggered and obs.get("dwell_time", 0) > policy.dwell_threshold_sec + 8) else 0.0
        spam_proxy = 1.0 if obs.get("zone_alert_rate", 0.0) > 4.0 else 0.0
        rec = {
            "record_type": "decision",
            "policy_version": policy.policy_version,
            "ts": time.time(),
            "observation": obs,
            "agent_output": out.model_dump(),
            "outcome_signals": {
                "resolved_proxy": resolved_proxy,
                "abandoned_proxy": abandoned_proxy,
                "spam_proxy": spam_proxy,
                "audio_feedback_proxy": 0.0,
            },
            "scores": {},
            "metadata": {
                "event_id": obs.get("event_id"),
                "alert_id": None,
                "zone": obs.get("zone"),
            },
        }
        rec = _evaluate_and_log_record(braintrust, rec)
        eval_records.append(rec)
        zone = str(obs.get("zone", "Unknown"))
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        zone_conf_proxy[zone] = zone_conf_proxy.get(zone, 0.0) + (1.0 - spam_proxy) * 0.1

    # Simulate periodic RCI evaluation records in simulation mode.
    policy = policy_store.load()
    for zone, cnt in zone_counts.items():
        if cnt < 3:
            continue
        avg_conf = round(zone_conf_proxy.get(zone, 0.0) / max(cnt, 1), 3)
        rci_rec = {
            "record_type": "rci",
            "policy_version": policy.policy_version,
            "ts": time.time(),
            "observation": {
                "zone": zone,
                "alerts_count": cnt,
                "avg_confidence": avg_conf,
                "avg_dwell": 24.0,
                "abandon_rate_proxy": 0.2,
                "queue_stats": round(cnt / 5.0, 2),
            },
            "agent_output": {
                "issue": f"Clustered assistance demand in {zone}",
                "confidence": min(1.0, 0.55 + cnt * 0.03),
                "recommended_action": "Assign staff to zone and restock top confusion SKUs",
                "reason": "Repeated assistance candidates within evaluation window",
            },
            "outcome_signals": {
                "resolved_proxy": 0.0,
                "abandoned_proxy": 0.0,
                "spam_proxy": 0.0,
                "audio_feedback_proxy": 0.0,
            },
            "scores": {},
            "metadata": {
                "event_id": None,
                "alert_id": None,
                "zone": zone,
            },
        }
        rci_eval = _evaluate_and_log_record(braintrust, rci_rec)
        eval_records.append(rci_eval)

    if len(eval_records) >= SETTINGS.optimize_every_n_events:
        limit = max(50, SETTINGS.optimize_every_n_events)
        bt_records = braintrust.fetch_recent_eval_records(limit=limit)
        window = bt_records if len(bt_records) >= SETTINGS.optimize_every_n_events else eval_records[-SETTINGS.optimize_every_n_events :]
        optimizer.optimize(window)

    metrics_store.write(_compute_metrics(eval_records))
    LOGGER.info("Simulation complete. eval_records=%d", len(eval_records))


def main() -> None:
    ensure_runtime_files()

    gemini = GeminiClient(api_key=SETTINGS.gemini_api_key, model=SETTINGS.gemini_model)
    decision_agent = DecisionAgent(gemini)
    rci_agent = RootCauseAgent(gemini)

    policy_store = PolicyStore(
        policy_path=SETTINGS.runtime_dir / "policy.json",
        changes_path=SETTINGS.runtime_dir / "policy_changes.jsonl",
    )
    optimizer = OptimizationAgent(policy_store)
    metrics_store = MetricsStore(SETTINGS.runtime_dir / "metrics.json")

    braintrust = BraintrustClient(
        api_key=SETTINGS.braintrust_api_key,
        project=SETTINGS.braintrust_project,
        local_path=SETTINGS.runtime_dir / "braintrust_log.jsonl",
    )

    voice = VoiceNotifier(
        client=ModulateClient(SETTINGS.modulate_api_key, SETTINGS.modulate_tts_url),
        enabled=SETTINGS.voice_enabled,
    )
    stt_client = ModulateSTTClient(
        api_key=SETTINGS.modulate_api_key,
        endpoint=SETTINGS.modulate_stt_url,
        enabled=SETTINGS.modulate_stt_enabled,
    )
    stt_payload = None
    stt_text = ""
    audio_proxy = 0.0
    if SETTINGS.modulate_stt_audio_path:
        stt_payload = stt_client.transcribe_file(Path(SETTINGS.modulate_stt_audio_path))
        if isinstance(stt_payload, dict):
            stt_text = str(stt_payload.get("text", "")).strip()
            if stt_text:
                audio_proxy = _audio_feedback_proxy(stt_text)

    if SETTINGS.simulation_mode:
        _run_simulation(decision_agent, optimizer, policy_store, braintrust, metrics_store)
        return

    zone_mapper = ZoneMapper(SETTINGS.zones)
    tracker = MultiObjectTracker(SETTINGS.yolo_model, SETTINGS.yolo_conf)
    state_store = PersonStateStore()
    aggregator = RCIAggregator(SETTINGS.rci_window_sec, SETTINGS.rci_min_alerts, SETTINGS.rci_cooldown_sec)

    cap = cv2.VideoCapture(_video_source(SETTINGS.video_source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {SETTINGS.video_source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(int(src_fps / max(SETTINGS.run_fps, 1)), 1)

    last_event_check = 0.0
    last_rci_check = 0.0
    frame_index = 0
    last_alert_text = "none"

    alerts: deque[dict] = deque(maxlen=500)
    rci_recs: deque[dict] = deque(maxlen=200)
    eval_records: list[dict] = []

    pending_outcomes: list[dict[str, Any]] = []

    LOGGER.info("ShelfSense started source=%s frame_step=%d", SETTINGS.video_source, frame_step)

    while True:
        ok, frame = cap.read()
        if not ok:
            LOGGER.info("Video ended or read failed")
            if pending_outcomes:
                policy = policy_store.load()
                now_final = time.time()
                active = {p.person_id: p for p in state_store.active_people()}
                for item in pending_outcomes:
                    alert = item["alert"]
                    obs = item["observation"]
                    decision_out = item["decision"]

                    person_id = alert["person_id"]
                    person = active.get(person_id)

                    resolved_proxy = 1.0 if person is None or person.current_zone != alert["zone"] else 0.0
                    abandoned_proxy = 1.0 if person is None and obs.get("dwell_time", 0.0) >= policy.dwell_threshold_sec + 5 else 0.0
                    zone_recent = [a for a in alerts if a["zone"] == alert["zone"] and (now_final - a["ts"] <= 60)]
                    spam_proxy = 1.0 if len(zone_recent) >= 5 else 0.0

                    eval_rec = {
                        "record_type": "decision",
                        "policy_version": policy.policy_version,
                        "ts": now_final,
                        "observation": obs,
                        "agent_output": decision_out,
                        "outcome_signals": {
                            "resolved_proxy": resolved_proxy,
                            "abandoned_proxy": abandoned_proxy,
                            "spam_proxy": spam_proxy,
                            "audio_feedback_proxy": audio_proxy,
                        },
                        "scores": {},
                        "metadata": {
                            "event_id": alert["event_id"],
                            "alert_id": alert["alert_id"],
                            "zone": alert["zone"],
                            "audio_feedback_excerpt": stt_text[:120] if stt_text else "",
                            "finalized_at_video_end": True,
                        },
                    }
                    eval_rec = _evaluate_and_log_record(braintrust, eval_rec, sink_path=SETTINGS.runtime_dir / "outcomes.jsonl")
                    eval_records.append(eval_rec)
                pending_outcomes = []

                if len(eval_records) >= SETTINGS.optimize_every_n_events:
                    limit = max(50, SETTINGS.optimize_every_n_events)
                    bt_records = braintrust.fetch_recent_eval_records(limit=limit)
                    window = bt_records if len(bt_records) >= SETTINGS.optimize_every_n_events else eval_records[-SETTINGS.optimize_every_n_events :]
                    optimizer.optimize(window)
            break
        frame_index += 1
        now = _stream_time_seconds(cap, SETTINGS.video_source)

        if frame_index % frame_step != 0:
            continue

        policy = policy_store.load()
        if SETTINGS.policy_override_from_env:
            policy.alert_conf_threshold = SETTINGS.alert_conf_threshold
            policy.dwell_threshold_sec = SETTINGS.dwell_threshold_sec
            policy.motion_threshold = SETTINGS.motion_threshold
            policy.rci_min_alerts = SETTINGS.rci_min_alerts
        policy_dict = asdict(policy)
        event_trigger = EventTrigger(
            dwell_threshold_sec=policy.dwell_threshold_sec,
            motion_threshold=policy.motion_threshold,
        )

        tracks = tracker.update(frame)
        h, w = frame.shape[:2]
        overlay_tracks: list[dict] = []

        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            zone = zone_mapper.bbox_to_zone(tr.bbox, w, h)
            cx, cy = ((x1 + x2) / 2.0) / max(w, 1), ((y1 + y2) / 2.0) / max(h, 1)
            ps = state_store.update_person(tr.person_id, zone, (cx, cy), now)
            overlay_tracks.append(
                {
                    "person_id": tr.person_id,
                    "bbox": tr.bbox,
                    "zone": ps.current_zone,
                    "dwell": ps.dwell_time_sec,
                    "motion": ps.motion_score,
                }
            )

        state_store.prune_stale(SETTINGS.track_lost_timeout_sec, now_ts=now)

        if now - last_event_check >= SETTINGS.event_check_interval_sec:
            last_event_check = now
            active_people = state_store.active_people()
            for ps in active_people:
                event = event_trigger.maybe_trigger(ps)
                if event is None:
                    continue

                observation = event.to_observation()
                append_jsonl(SETTINGS.runtime_dir / "events.jsonl", observation)

                out = decision_agent.decide(observation)
                out_dict = out.model_dump()

                should_alert = out.alert and out.confidence >= policy.alert_conf_threshold
                if should_alert:
                    alert_id = str(uuid.uuid4())
                    alert_rec = {
                        "alert_id": alert_id,
                        "event_id": event.event_id,
                        "ts": now,
                        "zone": event.zone,
                        "confidence": out.confidence,
                        "action": out.recommended_action,
                        "reason": out.reason,
                        "tags": out.tags,
                        "person_id": event.person_id,
                        "dwell_time": event.dwell_time,
                    }
                    alerts.append(alert_rec)
                    append_jsonl(SETTINGS.runtime_dir / "alerts.jsonl", alert_rec)
                    state_store.mark_alert_sent(event.person_id, event.zone)
                    aggregator.add_alert(
                        AlertRecord(
                            alert_id=alert_id,
                            event_id=event.event_id,
                            ts=now,
                            zone=event.zone,
                            confidence=out.confidence,
                            dwell_time=event.dwell_time,
                        )
                    )
                    voice.speak(f"Shelf assistance suggested at {event.zone}. {out.recommended_action}")
                    last_alert_text = f"{event.zone} {out.confidence:.2f}"

                    pending_outcomes.append(
                        {
                            "due_ts": now + SETTINGS.outcome_eval_delay_sec,
                            "alert": alert_rec,
                            "decision": out_dict,
                            "observation": observation,
                        }
                    )

        if now - last_rci_check >= 30.0:
            last_rci_check = now
            clusters = aggregator.cluster_candidates()
            for cluster in clusters:
                rci_out = rci_agent.analyze(cluster)
                rec = {
                    "ts": now,
                    "zone": cluster["zone"],
                    "issue": rci_out.issue,
                    "confidence": rci_out.confidence,
                    "recommended_action": rci_out.recommended_action,
                    "reason": rci_out.reason,
                    "cluster": cluster,
                }
                rci_recs.append(rec)
                append_jsonl(SETTINGS.runtime_dir / "rci.jsonl", rec)
                rci_eval_rec = {
                    "record_type": "rci",
                    "policy_version": policy.policy_version,
                    "ts": now,
                    "observation": cluster,
                    "agent_output": {
                        "issue": rci_out.issue,
                        "confidence": rci_out.confidence,
                        "recommended_action": rci_out.recommended_action,
                        "reason": rci_out.reason,
                    },
                    "outcome_signals": {
                        "resolved_proxy": 0.0,
                        "abandoned_proxy": 0.0,
                        "spam_proxy": 0.0,
                        "audio_feedback_proxy": audio_proxy,
                    },
                    "scores": {},
                    "metadata": {
                        "event_id": None,
                        "alert_id": None,
                        "zone": cluster["zone"],
                        "audio_feedback_excerpt": stt_text[:120] if stt_text else "",
                    },
                }
                rci_eval_rec = _evaluate_and_log_record(braintrust, rci_eval_rec)
                eval_records.append(rci_eval_rec)
                voice.speak(f"Root cause detected in {cluster['zone']}. {rci_out.recommended_action}")

        # Evaluate outcomes 30-90 sec later, no human feedback.
        unresolved: list[dict[str, Any]] = []
        for item in pending_outcomes:
            if now < item["due_ts"]:
                unresolved.append(item)
                continue

            alert = item["alert"]
            obs = item["observation"]
            decision_out = item["decision"]

            active = {p.person_id: p for p in state_store.active_people()}
            person_id = alert["person_id"]
            person = active.get(person_id)

            resolved_proxy = 1.0 if person is None or person.current_zone != alert["zone"] else 0.0
            abandoned_proxy = 1.0 if person is None and obs.get("dwell_time", 0.0) >= policy.dwell_threshold_sec + 5 else 0.0

            zone_recent = [a for a in alerts if a["zone"] == alert["zone"] and (now - a["ts"] <= 60)]
            spam_proxy = 1.0 if len(zone_recent) >= 5 else 0.0

            eval_rec = {
                "record_type": "decision",
                "policy_version": policy.policy_version,
                "ts": now,
                "observation": obs,
                "agent_output": decision_out,
                "outcome_signals": {
                    "resolved_proxy": resolved_proxy,
                    "abandoned_proxy": abandoned_proxy,
                    "spam_proxy": spam_proxy,
                    "audio_feedback_proxy": audio_proxy,
                },
                "scores": {},
                "metadata": {
                    "event_id": alert["event_id"],
                    "alert_id": alert["alert_id"],
                    "zone": alert["zone"],
                    "audio_feedback_excerpt": stt_text[:120] if stt_text else "",
                },
            }
            eval_rec = _evaluate_and_log_record(braintrust, eval_rec, sink_path=SETTINGS.runtime_dir / "outcomes.jsonl")
            eval_records.append(eval_rec)

            if len(eval_records) % SETTINGS.optimize_every_n_events == 0:
                limit = max(50, SETTINGS.optimize_every_n_events)
                bt_records = braintrust.fetch_recent_eval_records(limit=limit)
                window = bt_records if len(bt_records) >= SETTINGS.optimize_every_n_events else eval_records[-SETTINGS.optimize_every_n_events :]
                optimizer.optimize(window)

        pending_outcomes = unresolved

        metrics_store.write(_compute_metrics(eval_records))
        _update_runtime_state(policy_dict, list(alerts), list(rci_recs))

        _draw_overlay(frame, overlay_tracks, zone_mapper, policy_dict, last_alert_text)
        if SETTINGS.show_debug_window:
            cv2.imshow("ShelfSense AI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SETTINGS.show_debug_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
