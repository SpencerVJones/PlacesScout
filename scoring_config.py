"""Shared scoring configuration and weight parsing utilities."""

from __future__ import annotations

import math
from typing import Dict

DEFAULT_WEIGHTS = {
    "affordability": 0.30,
    "safety": 0.35,
    "transit": 0.20,
    "amenities": 0.15,
}

WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS.keys())


def parse_weights(raw: str) -> Dict[str, float]:
    if not raw.strip():
        return DEFAULT_WEIGHTS.copy()

    values: Dict[str, float] = {}
    for piece in raw.split(","):
        if "=" not in piece:
            raise ValueError(f"Invalid weight format: '{piece}'. Use key=value.")
        key, value = piece.split("=", maxsplit=1)
        key = key.strip().lower()
        if key not in WEIGHT_KEYS:
            raise ValueError(
                f"Unknown weight key: '{key}'. Valid keys: {', '.join(WEIGHT_KEYS)}."
            )
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Weight for '{key}' is not numeric.") from exc
        if number < 0:
            raise ValueError(f"Weight for '{key}' cannot be negative.")
        values[key] = number

    for key in WEIGHT_KEYS:
        values.setdefault(key, DEFAULT_WEIGHTS[key])

    total = sum(values.values())
    if math.isclose(total, 0.0):
        raise ValueError("At least one weight must be greater than zero.")
    return {key: values[key] / total for key in WEIGHT_KEYS}
