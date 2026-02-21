from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from config import append_jsonl

LOGGER = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = {
    "face",
    "identity",
    "emotion",
    "male",
    "female",
    "race",
    "ethnicity",
    "age",
}


class BraintrustClient:
    """
    Braintrust-backed evaluation client.

    It evaluates records (decision/rci), stores a stable unified schema locally,
    and uses Braintrust as the primary eval backend when configured.
    """

    def __init__(self, api_key: str, project: str, local_path: Path) -> None:
        self.api_key = api_key
        self.project = project
        self.local_path = local_path
        self.enabled = bool(api_key and project)
        self.base_url = "https://api.braintrust.dev"
        self._sdk_ready = False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _tail_local(self, limit: int) -> list[dict[str, Any]]:
        if not self.local_path.exists():
            return []
        lines = self.local_path.read_text(encoding="utf-8").strip().splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _decision_output_is_valid(self, output: dict[str, Any]) -> bool:
        try:
            if not isinstance(output.get("alert"), bool):
                return False
            conf = float(output.get("confidence", -1.0))
            if conf < 0.0 or conf > 1.0:
                return False
            if not isinstance(output.get("recommended_action", ""), str):
                return False
            if not isinstance(output.get("reason", ""), str):
                return False
            tags = output.get("tags", [])
            if not isinstance(tags, list):
                return False
            return all(isinstance(t, str) for t in tags)
        except Exception:
            return False

    def _rci_output_is_valid(self, output: dict[str, Any]) -> bool:
        try:
            if not isinstance(output.get("issue", ""), str):
                return False
            conf = float(output.get("confidence", -1.0))
            if conf < 0.0 or conf > 1.0:
                return False
            if not isinstance(output.get("recommended_action", ""), str):
                return False
            if not isinstance(output.get("reason", ""), str):
                return False
            return True
        except Exception:
            return False

    def _output_is_valid(self, record_type: str, output: dict[str, Any]) -> bool:
        if record_type == "rci":
            return self._rci_output_is_valid(output)
        return self._decision_output_is_valid(output)

    def _forbidden_penalty(self, output: dict[str, Any]) -> float:
        text = " ".join(
            [
                str(output.get("issue", "")),
                str(output.get("reason", "")),
                str(output.get("recommended_action", "")),
            ]
        ).lower()
        return 1.0 if any(k in text for k in FORBIDDEN_KEYWORDS) else 0.0

    def _overall_raw(
        self,
        resolved_proxy: float,
        abandoned_proxy: float,
        spam_proxy: float,
        audio_feedback_proxy: float,
        invalid_json_penalty: float,
        forbidden_content_penalty: float,
    ) -> float:
        return round(
            resolved_proxy
            - abandoned_proxy
            - 0.8 * spam_proxy
            + 0.3 * audio_feedback_proxy
            - invalid_json_penalty
            - forbidden_content_penalty,
            4,
        )

    def _normalize_overall(self, overall_raw: float) -> float:
        # raw theoretical range is approximately [-2.8, 1.0]
        return round(max(0.0, min(1.0, (overall_raw + 2.8) / 3.8)), 4)

    def _ensure_sdk(self) -> bool:
        if not self.enabled:
            return False
        if self._sdk_ready:
            return True
        try:
            import braintrust

            braintrust.init_logger(project=self.project, api_key=self.api_key, set_current=True)
            self._sdk_ready = True
            return True
        except Exception as exc:
            LOGGER.warning("Braintrust SDK init failed: %s", exc)
            return False

    def _ensure_record_schema(self, record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        out.setdefault("record_type", "decision")
        out.setdefault("policy_version", 1)
        out.setdefault("ts", time.time())
        out.setdefault("observation", {})
        out.setdefault("agent_output", {})
        out.setdefault("outcome_signals", {})
        out.setdefault("scores", {})
        out.setdefault("metadata", {})
        return out

    def evaluate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a unified schema record with Braintrust scorers.

        Required shape:
          record_type, policy_version, ts, observation, agent_output,
          outcome_signals, scores, metadata
        """
        rec = self._ensure_record_schema(record)
        if not self._ensure_sdk():
            rec["_braintrust_eval_success"] = False
            return rec

        try:
            from braintrust import Eval, Score
            from braintrust.git_fields import GitMetadataSettings

            case = {
                "input": rec["observation"],
                "expected": rec["outcome_signals"],
                "metadata": {
                    "record_type": rec["record_type"],
                    "policy_version": rec["policy_version"],
                    "ts": float(rec["ts"]),
                    "agent_output": rec["agent_output"],
                    "metadata": rec["metadata"],
                },
            }

            def task(_input: dict[str, Any], hooks: Any) -> dict[str, Any]:
                return (hooks.metadata or {}).get("agent_output", {})

            def score_resolved(_input: dict[str, Any], _output: dict[str, Any], expected: dict[str, Any]) -> Score:
                return Score(name="resolved_proxy", score=float(expected.get("resolved_proxy", 0.0)))

            def score_abandoned(_input: dict[str, Any], _output: dict[str, Any], expected: dict[str, Any]) -> Score:
                return Score(name="abandoned_proxy", score=float(expected.get("abandoned_proxy", 0.0)))

            def score_spam(_input: dict[str, Any], _output: dict[str, Any], expected: dict[str, Any]) -> Score:
                return Score(name="spam_proxy", score=float(expected.get("spam_proxy", 0.0)))

            def score_audio_feedback(_input: dict[str, Any], _output: dict[str, Any], expected: dict[str, Any]) -> Score:
                val = float(expected.get("audio_feedback_proxy", 0.0))
                val = max(0.0, min(1.0, val))
                return Score(name="audio_feedback_proxy", score=val)

            def score_invalid_json(_input: dict[str, Any], output: dict[str, Any], _expected: dict[str, Any], hooks: Any = None) -> Score:
                record_type = "decision"
                if hooks is not None and getattr(hooks, "metadata", None):
                    record_type = hooks.metadata.get("record_type", "decision")
                pen = 0.0 if self._output_is_valid(record_type, output) else 1.0
                return Score(name="invalid_json_penalty", score=pen)

            def score_forbidden(_input: dict[str, Any], output: dict[str, Any], _expected: dict[str, Any]) -> Score:
                pen = self._forbidden_penalty(output)
                return Score(name="forbidden_content_penalty", score=pen)

            def score_overall(_input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], hooks: Any = None) -> Score:
                record_type = "decision"
                if hooks is not None and getattr(hooks, "metadata", None):
                    record_type = hooks.metadata.get("record_type", "decision")
                invalid = 0.0 if self._output_is_valid(record_type, output) else 1.0
                forbidden = self._forbidden_penalty(output)
                raw = self._overall_raw(
                    resolved_proxy=float(expected.get("resolved_proxy", 0.0)),
                    abandoned_proxy=float(expected.get("abandoned_proxy", 0.0)),
                    spam_proxy=float(expected.get("spam_proxy", 0.0)),
                    audio_feedback_proxy=float(expected.get("audio_feedback_proxy", 0.0)),
                    invalid_json_penalty=invalid,
                    forbidden_content_penalty=forbidden,
                )
                return Score(name="overall", score=self._normalize_overall(raw))

            result = Eval(
                name="ShelfSense AI Eval",
                experiment_name="autonomous-runtime",
                data=[case],
                task=task,
                scores=[
                    score_resolved,
                    score_abandoned,
                    score_spam,
                    score_audio_feedback,
                    score_invalid_json,
                    score_forbidden,
                    score_overall,
                ],
                summarize_scores=False,
                max_concurrency=1,
                timeout=15,
                git_metadata_settings=GitMetadataSettings(collect="none"),
            )
            row = result.results[0]
            s = row.scores
            invalid = 0.0 if self._output_is_valid(rec["record_type"], rec["agent_output"]) else 1.0
            forbidden = self._forbidden_penalty(rec["agent_output"])
            raw = self._overall_raw(
                resolved_proxy=float(rec["outcome_signals"].get("resolved_proxy", 0.0)),
                abandoned_proxy=float(rec["outcome_signals"].get("abandoned_proxy", 0.0)),
                spam_proxy=float(rec["outcome_signals"].get("spam_proxy", 0.0)),
                audio_feedback_proxy=float(rec["outcome_signals"].get("audio_feedback_proxy", 0.0)),
                invalid_json_penalty=invalid,
                forbidden_content_penalty=forbidden,
            )
            rec["scores"] = {
                "overall_raw": raw,
                "overall": float(s.get("overall", 0.0)),
                "resolved_proxy": float(s.get("resolved_proxy", 0.0)),
                "abandoned_proxy": float(s.get("abandoned_proxy", 0.0)),
                "spam_proxy": float(s.get("spam_proxy", 0.0)),
                "audio_feedback_proxy": float(s.get("audio_feedback_proxy", 0.0)),
                "invalid_json_penalty": invalid,
                "forbidden_content_penalty": forbidden,
            }
            rec["_braintrust_eval_success"] = True
            rec["_braintrust_eval_logged"] = True
            return rec
        except Exception as exc:
            LOGGER.warning("Braintrust Eval failed; fallback scoring will be used. reason=%s", exc)
            rec["_braintrust_eval_success"] = False
            return rec

    def fallback_score_record(self, record: dict[str, Any]) -> dict[str, Any]:
        rec = self._ensure_record_schema(record)
        osig = rec["outcome_signals"]
        output = rec["agent_output"]
        invalid = 0.0 if self._output_is_valid(rec["record_type"], output) else 1.0
        forbidden = self._forbidden_penalty(output)
        raw = self._overall_raw(
            resolved_proxy=float(osig.get("resolved_proxy", 0.0)),
            abandoned_proxy=float(osig.get("abandoned_proxy", 0.0)),
            spam_proxy=float(osig.get("spam_proxy", 0.0)),
            audio_feedback_proxy=float(osig.get("audio_feedback_proxy", 0.0)),
            invalid_json_penalty=invalid,
            forbidden_content_penalty=forbidden,
        )
        rec["scores"] = {
            "overall_raw": raw,
            "overall": self._normalize_overall(raw),
            "resolved_proxy": float(osig.get("resolved_proxy", 0.0)),
            "abandoned_proxy": float(osig.get("abandoned_proxy", 0.0)),
            "spam_proxy": float(osig.get("spam_proxy", 0.0)),
            "audio_feedback_proxy": float(osig.get("audio_feedback_proxy", 0.0)),
            "invalid_json_penalty": invalid,
            "forbidden_content_penalty": forbidden,
        }
        return rec

    def log_record(self, record: dict[str, Any]) -> None:
        rec = self._ensure_record_schema(record)
        append_jsonl(self.local_path, rec)
        if not self.enabled:
            return

        # If this record already came from Braintrust Eval(), avoid duplicate API log calls.
        if rec.get("_braintrust_eval_logged"):
            return

        payload = {
            "project": self.project,
            "rows": [
                {
                    "id": rec["metadata"].get("alert_id") or rec["metadata"].get("event_id") or str(time.time_ns()),
                    "input": rec["observation"],
                    "output": rec["agent_output"],
                    "scores": {
                        "overall": float(rec["scores"].get("overall", 0.0)),
                        "resolved_proxy": float(rec["scores"].get("resolved_proxy", 0.0)),
                        "abandoned_proxy": float(rec["scores"].get("abandoned_proxy", 0.0)),
                        "spam_proxy": float(rec["scores"].get("spam_proxy", 0.0)),
                        "audio_feedback_proxy": float(rec["scores"].get("audio_feedback_proxy", 0.0)),
                        "invalid_json_penalty": float(rec["scores"].get("invalid_json_penalty", 0.0)),
                        "forbidden_content_penalty": float(rec["scores"].get("forbidden_content_penalty", 0.0)),
                    },
                    "metadata": {
                        "ts": rec["ts"],
                        "record_type": rec["record_type"],
                        "policy_version": rec["policy_version"],
                        "overall_raw": float(rec["scores"].get("overall_raw", 0.0)),
                        **rec["metadata"],
                    },
                    "tags": ["shelfsense", "autonomous-eval", rec["record_type"]],
                }
            ],
        }

        for attempt in range(2):
            try:
                resp = requests.post(f"{self.base_url}/v1/logs", headers=self._headers(), json=payload, timeout=8)
                if resp.status_code < 300:
                    return
            except Exception as exc:
                LOGGER.warning("Braintrust log request error: %s", exc)
            time.sleep(0.5 * (attempt + 1))

    def fetch_recent_eval_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Fetch recent eval records from Braintrust for optimization loop.
        Falls back to local records if remote query is unavailable.
        """
        if not self.enabled:
            return self._tail_local(limit)

        query_urls = [
            f"{self.base_url}/v1/logs/query",
            f"{self.base_url}/v1/logs/search",
        ]
        query_payloads = [
            {"project": self.project, "limit": limit, "order": "desc"},
            {"project_name": self.project, "limit": limit, "order": "desc"},
        ]

        for url in query_urls:
            for payload in query_payloads:
                try:
                    resp = requests.post(url, headers=self._headers(), json=payload, timeout=10)
                    if resp.status_code >= 300:
                        continue
                    data = resp.json()
                    rows = data.get("rows") or data.get("data") or data.get("results") or []
                    converted: list[dict[str, Any]] = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        md = row.get("metadata", {})
                        converted.append(
                            self._ensure_record_schema(
                                {
                                    "record_type": md.get("record_type", "decision"),
                                    "policy_version": int(md.get("policy_version", 1)),
                                    "ts": float(md.get("ts", time.time())),
                                    "observation": row.get("input", {}),
                                    "agent_output": row.get("output", {}),
                                    "outcome_signals": {
                                        "resolved_proxy": float(row.get("scores", {}).get("resolved_proxy", 0.0)),
                                        "abandoned_proxy": float(row.get("scores", {}).get("abandoned_proxy", 0.0)),
                                        "spam_proxy": float(row.get("scores", {}).get("spam_proxy", 0.0)),
                                        "audio_feedback_proxy": float(row.get("scores", {}).get("audio_feedback_proxy", 0.0)),
                                    },
                                    "scores": {
                                        "overall_raw": float(row.get("scores", {}).get("overall_raw", 0.0)),
                                        "overall": float(row.get("scores", {}).get("overall", 0.0)),
                                        "resolved_proxy": float(row.get("scores", {}).get("resolved_proxy", 0.0)),
                                        "abandoned_proxy": float(row.get("scores", {}).get("abandoned_proxy", 0.0)),
                                        "spam_proxy": float(row.get("scores", {}).get("spam_proxy", 0.0)),
                                        "audio_feedback_proxy": float(row.get("scores", {}).get("audio_feedback_proxy", 0.0)),
                                        "invalid_json_penalty": float(row.get("scores", {}).get("invalid_json_penalty", 0.0)),
                                        "forbidden_content_penalty": float(
                                            row.get("scores", {}).get("forbidden_content_penalty", 0.0)
                                        ),
                                    },
                                    "metadata": {
                                        "event_id": md.get("event_id"),
                                        "alert_id": md.get("alert_id"),
                                        "zone": md.get("zone"),
                                    },
                                }
                            )
                        )
                    if converted:
                        return converted[-limit:]
                except Exception as exc:
                    LOGGER.warning("Braintrust fetch request error: %s", exc)

        return self._tail_local(limit)
