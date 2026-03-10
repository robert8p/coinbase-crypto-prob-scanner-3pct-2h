from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def heuristic_prob_and_notes(row: pd.Series) -> Tuple[float, str, str]:
    """
    Deterministic fallback probability when no trained model exists.
    Returns: (prob, notes, risk_flag)
    """
    eps = 1e-12
    # Pull values defensively
    imp = float(row.get("impulse_60m", 0.0) or 0.0)
    acc = float(row.get("accel_30_60", 0.0) or 0.0)
    rv60 = float(row.get("rv_60m", 0.0) or 0.0)
    rv6h = float(row.get("rv_6h", 0.0) or 0.0)
    donch6 = float(row.get("donch_dist_6h", 0.0) or 0.0)
    bbw = float(row.get("bb_bw", 0.0) or 0.0)
    bbw_pct = float(row.get("bb_bw_pct", 1.0) or 1.0)
    btc_ret = float(row.get("btc_ret_30m", 0.0) or 0.0)
    btc_rv = float(row.get("btc_rv_6h", 0.0) or 0.0)

    # Core intuition:
    # - Positive normalized momentum helps
    # - Being close to a recent high helps
    # - Excessive 6h vol hurts (harder to sustain a directed +3% move without whipsaw)
    # - Very tight/very wide BB interpreted via bbw_pct
    z = 0.0
    z += 1.6 * imp
    z += 0.8 * (acc / (rv60 + eps))
    z += 0.7 * (1.0 - min(max(donch6, 0.0), 1.5))  # closer to high => larger
    z += 0.35 * (bbw_pct - 1.0)
    z -= 1.2 * rv6h

    # BTC regime: if BTC is trending up and not extremely volatile, modest boost
    z += 0.35 * (btc_ret / (btc_rv + 1e-6))

    # Convert to probability and temper to a realistic heuristic range
    p = _sigmoid(z)
    p = float(min(0.98, max(0.02, p)))

    notes = []
    if imp > 1.0:
        notes.append("MOMO_UP")
    if donch6 < 0.03:
        notes.append("NEAR_6H_HIGH")
    if bbw_pct > 1.3:
        notes.append("EXPANDING_RANGE")
    if rv6h > 0.01:
        notes.append("HIGH_VOL")
    if btc_ret > 0 and btc_rv < 0.01:
        notes.append("BTC_TAILWIND")

    # Simple risk flag
    risk = "OK"
    if rv6h > 0.02:
        risk = "CAUTION"
    if rv6h > 0.035:
        risk = "HIGH"

    return p, ",".join(notes) if notes else "heuristic", risk
