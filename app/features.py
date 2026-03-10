from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Momentum
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_60m",
    "impulse_60m",
    "accel_30_60",
    # Volatility
    "atr_pct",
    "rv_60m",
    "rv_6h",
    "range_pct",
    "range_vs_med",
    "vol_of_vol",
    # Breakout
    "donch_dist_6h",
    "donch_dist_24h",
    "bb_bw",
    "bb_bw_pct",
    # BTC regime
    "btc_ret_30m",
    "btc_rv_6h",
    # Time
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
]


def _safe_log_ret(a: pd.Series, b: pd.Series) -> pd.Series:
    eps = 1e-12
    return np.log((a + eps) / (b + eps))


def _rv(returns: pd.Series, window: int) -> pd.Series:
    # realized volatility proxy: sqrt(mean(r^2)) over window
    r2 = returns ** 2
    return np.sqrt(r2.rolling(window, min_periods=window).mean())


def _atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    eps = 1e-12
    return atr / (df["close"] + eps)


def _time_features(ts_seconds: pd.Series) -> pd.DataFrame:
    # ts_seconds: candle start epoch seconds (UTC)
    dt = pd.to_datetime(ts_seconds, unit="s", utc=True)
    minutes = dt.dt.hour * 60 + dt.dt.minute
    tod = (minutes / 1440.0).astype(float)
    tod_sin = np.sin(2 * np.pi * tod)
    tod_cos = np.cos(2 * np.pi * tod)

    dow = (dt.dt.dayofweek / 7.0).astype(float)
    dow_sin = np.sin(2 * np.pi * dow)
    dow_cos = np.cos(2 * np.pi * dow)

    return pd.DataFrame({
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    })


def compute_benchmark_regime(bench_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with columns: ts, btc_ret_30m, btc_rv_6h
    """
    b = bench_df.sort_values("ts").copy()
    b["ret_5m_b"] = _safe_log_ret(b["close"], b["close"].shift(1))
    b["btc_ret_30m"] = _safe_log_ret(b["close"], b["close"].shift(6))
    b["btc_rv_6h"] = _rv(b["ret_5m_b"], 72)
    out = b[["ts", "btc_ret_30m", "btc_rv_6h"]].copy()
    return out


def compute_features(df: pd.DataFrame, bench_regime: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute the required feature schema on 5-minute candles.

    df columns required: ts, low, high, open, close, volume
    """
    x = df.sort_values("ts").copy()

    # Momentum (log returns)
    x["ret_5m"] = _safe_log_ret(x["close"], x["close"].shift(1))
    x["ret_15m"] = _safe_log_ret(x["close"], x["close"].shift(3))
    x["ret_30m"] = _safe_log_ret(x["close"], x["close"].shift(6))
    x["ret_60m"] = _safe_log_ret(x["close"], x["close"].shift(12))

    # Volatility core
    x["rv_60m"] = _rv(x["ret_5m"], 12)
    x["rv_6h"] = _rv(x["ret_5m"], 72)
    x["atr_pct"] = _atr_pct(x, 14)

    eps = 1e-12
    x["impulse_60m"] = x["ret_60m"] / (x["rv_60m"] + eps)
    x["accel_30_60"] = x["ret_30m"] - x["ret_60m"]

    x["range_pct"] = (x["high"] - x["low"]).abs() / (x["close"] + eps)
    med_range = x["range_pct"].rolling(72, min_periods=72).median()
    x["range_vs_med"] = x["range_pct"] / (med_range + eps)

    x["vol_of_vol"] = x["rv_60m"].rolling(72, min_periods=72).std()

    # Breakout
    donch_6h = x["high"].rolling(72, min_periods=72).max()
    donch_24h = x["high"].rolling(288, min_periods=288).max()
    x["donch_dist_6h"] = (donch_6h - x["close"]) / (x["close"] + eps)
    x["donch_dist_24h"] = (donch_24h - x["close"]) / (x["close"] + eps)

    ma = x["close"].rolling(20, min_periods=20).mean()
    sd = x["close"].rolling(20, min_periods=20).std()
    upper = ma + 2.0 * sd
    lower = ma - 2.0 * sd
    x["bb_bw"] = (upper - lower) / (ma + eps)
    bb_med = x["bb_bw"].rolling(288, min_periods=288).median()
    x["bb_bw_pct"] = x["bb_bw"] / (bb_med + eps)

    # Time features
    tf = _time_features(x["ts"])
    x = pd.concat([x.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)

    # BTC regime (as-of join on ts)
    if bench_regime is not None and not bench_regime.empty:
        b = bench_regime.sort_values("ts")
        # merge_asof expects sorted keys
        x = pd.merge_asof(
            x.sort_values("ts"),
            b,
            on="ts",
            direction="backward",
            tolerance=300,  # 5 minutes
        )
    else:
        x["btc_ret_30m"] = np.nan
        x["btc_rv_6h"] = np.nan

    # Ensure schema
    for col in FEATURE_COLUMNS:
        if col not in x.columns:
            x[col] = np.nan

    return x


def latest_feature_row(feat_df: pd.DataFrame) -> Optional[pd.Series]:
    if feat_df is None or feat_df.empty:
        return None
    return feat_df.iloc[-1]
