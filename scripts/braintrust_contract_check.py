from __future__ import annotations

import json
import sys
from pathlib import Path


RUNTIME_LOG = Path("runtime/braintrust_log.jsonl")


def main() -> int:
    if not RUNTIME_LOG.exists():
        print("FAIL: missing runtime/braintrust_log.jsonl")
        return 1

    rows = []
    for i, line in enumerate(RUNTIME_LOG.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"FAIL: invalid JSONL at line {i}")
            return 1

    if not rows:
        print("FAIL: no records in runtime/braintrust_log.jsonl")
        return 1

    types_seen: set[str] = set()
    for idx, r in enumerate(rows):
        if "policy_version" not in r:
            print(f"FAIL: record {idx} missing policy_version")
            return 1

        rtype = r.get("record_type")
        if rtype not in {"decision", "rci"}:
            print(f"FAIL: record {idx} invalid record_type={rtype}")
            return 1
        types_seen.add(rtype)

        if "outcome_signals" not in r or not isinstance(r["outcome_signals"], dict):
            print(f"FAIL: record {idx} missing outcome_signals dict")
            return 1

        scores = r.get("scores")
        if not isinstance(scores, dict):
            print(f"FAIL: record {idx} missing scores breakdown")
            return 1

        overall = scores.get("overall")
        if overall is None:
            print(f"FAIL: record {idx} missing scores.overall")
            return 1
        try:
            overall_f = float(overall)
        except Exception:
            print(f"FAIL: record {idx} scores.overall non-numeric")
            return 1
        if not (0.0 <= overall_f <= 1.0):
            print(f"FAIL: record {idx} scores.overall out of [0,1]: {overall_f}")
            return 1

    if not ({"decision", "rci"} <= types_seen):
        print(f"FAIL: missing record types. seen={sorted(types_seen)}")
        return 1

    print(f"PASS: {len(rows)} records validated. types={sorted(types_seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
