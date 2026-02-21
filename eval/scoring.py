from __future__ import annotations

from .datasets import FORBIDDEN_KEYWORDS


def has_forbidden_content(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in FORBIDDEN_KEYWORDS)


def compute_score(
    resolved_proxy: float,
    abandoned_proxy: float,
    spam_proxy: float,
    invalid_json_penalty: float,
    forbidden_content_penalty: float,
) -> float:
    score = (
        1.0 * resolved_proxy
        - 1.0 * abandoned_proxy
        - 0.8 * spam_proxy
        - invalid_json_penalty
        - forbidden_content_penalty
    )
    return round(score, 4)
