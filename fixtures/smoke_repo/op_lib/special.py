from __future__ import annotations

import math


def expit(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
