from __future__ import annotations

from optimize.policy_store import Policy, PolicyStore


class OptimizationAgent:
    def __init__(self, store: PolicyStore) -> None:
        self._store = store

    def optimize(self, recent_eval_records: list[dict]) -> Policy:
        policy = self._store.load()
        if not recent_eval_records:
            return policy

        avg_overall = sum(float(r.get("scores", {}).get("overall", 0.0)) for r in recent_eval_records) / len(recent_eval_records)

        decision_records = [r for r in recent_eval_records if r.get("record_type") == "decision"]
        if not decision_records:
            decision_records = recent_eval_records

        spam_rate = (
            sum(float(r.get("outcome_signals", {}).get("spam_proxy", 0.0)) for r in decision_records)
            / len(decision_records)
        )
        abandoned_rate = (
            sum(float(r.get("outcome_signals", {}).get("abandoned_proxy", 0.0)) for r in decision_records)
            / len(decision_records)
        )
        alert_rate = (
            sum(1.0 if r.get("agent_output", {}).get("alert") else 0.0 for r in decision_records)
            / len(decision_records)
        )

        reason_bits: list[str] = []
        changed = False

        if spam_rate > 0.2:
            policy.alert_conf_threshold = min(0.95, round(policy.alert_conf_threshold + 0.03, 3))
            policy.dwell_threshold_sec = min(60.0, round(policy.dwell_threshold_sec + 1.0, 2))
            reason_bits.append("spam_high")
            changed = True

        if abandoned_rate > 0.25 and alert_rate < 0.4:
            policy.dwell_threshold_sec = max(8.0, round(policy.dwell_threshold_sec - 1.0, 2))
            reason_bits.append("abandon_high_alert_low")
            changed = True

        # Braintrust overall trend steers sensitivity globally.
        if avg_overall < 0.45:
            policy.alert_conf_threshold = min(0.95, round(policy.alert_conf_threshold + 0.02, 3))
            policy.dwell_threshold_sec = min(60.0, round(policy.dwell_threshold_sec + 1.0, 2))
            reason_bits.append("overall_low")
            changed = True
        elif avg_overall > 0.72:
            policy.alert_conf_threshold = max(0.55, round(policy.alert_conf_threshold - 0.01, 3))
            reason_bits.append("overall_high")
            changed = True

        if not reason_bits:
            reason_bits.append("no_change")
        if changed:
            policy.policy_version += 1

        self._store.save(policy, reason=",".join(reason_bits))
        return policy
