from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import CandleStore, JsonStore
from .config import Settings
from .features import FEATURE_COLUMNS, compute_benchmark_regime, compute_features
from .modeling import ModelBundle, save_bundle, train_logreg_elasticnet_calibrated
from .scheduler import GRANULARITY_SEC, _candles_to_df, iso_z, floor_to_5m


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def fetch_candles_full_lookback(
    client,
    store: CandleStore,
    product_id: str,
    end_dt: datetime,
    lookback_days: int,
    progress_cb=None,
) -> Optional[pd.DataFrame]:
    """
    Fetches all candles needed for a full lookback window.
    Uses disk cache; fetches only missing portions at the beginning/end of the window.
    """
    end_dt = floor_to_5m(end_dt)
    start_dt = floor_to_5m(end_dt - timedelta(days=lookback_days))

    existing = store.load(product_id)
    have_start = None
    have_end = None
    if existing is not None and not existing.empty:
        have_start = int(existing["ts"].min())
        have_end = int(existing["ts"].max())

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    max_span_sec = GRANULARITY_SEC * 300  # Coinbase max

    def make_chunks(a: int, b: int):
        chunks = []
        cur = a
        while cur < b:
            chunk_end = min(cur + max_span_sec, b)
            chunks.append((cur, chunk_end))
            cur = chunk_end
        return chunks

    chunks = []
    if have_start is None or have_end is None:
        chunks = make_chunks(start_ts, end_ts)
    else:
        # fetch older part if needed
        if have_start > start_ts:
            chunks.extend(make_chunks(start_ts, min(have_start, end_ts)))
        # fetch newer part if needed
        if have_end + GRANULARITY_SEC < end_ts:
            chunks.extend(make_chunks(have_end + GRANULARITY_SEC, end_ts))

    if not chunks:
        return existing

    all_new = []
    for idx, (s_ts, e_ts) in enumerate(chunks):
        s_iso = iso_z(datetime.fromtimestamp(s_ts, tz=timezone.utc))
        e_iso = iso_z(datetime.fromtimestamp(e_ts, tz=timezone.utc))
        rows = await client.candles(product_id, s_iso, e_iso, granularity=GRANULARITY_SEC)
        df_new = _candles_to_df(rows)
        if not df_new.empty:
            all_new.append(df_new)
        if progress_cb:
            progress_cb({"product": product_id, "chunk": idx + 1, "chunks": len(chunks)})

    if all_new:
        new_df = pd.concat(all_new, ignore_index=True)
        merged = store.upsert_rows(product_id, new_df)
        return merged
    return existing


def build_labels_and_features_for_product(
    df: pd.DataFrame,
    bench_regime: Optional[pd.DataFrame],
    target_pct: float,
    horizon_minutes: int,
    decision_every_n_5m: int,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns: ts, y + FEATURE_COLUMNS for each decision timestamp.
    """
    horizon_bars = int(horizon_minutes // 5)
    if horizon_bars < 1:
        raise ValueError("HORIZON_MINUTES must be >= 5")

    feat = compute_features(df, bench_regime=bench_regime)
    feat = feat.reset_index(drop=True)

    # decision points: every N 5m bars, aligned to bar close (we use candle close)
    idxs = list(range(0, len(feat), max(1, int(decision_every_n_5m))))

    y = np.full(len(feat), np.nan, dtype=float)
    closes = feat["close"].to_numpy(dtype=float)
    highs = feat["high"].to_numpy(dtype=float)

    for i in idxs:
        j0 = i + 1
        j1 = i + 1 + horizon_bars
        if j1 > len(feat):
            continue
        p0 = closes[i]
        h_future = float(np.max(highs[j0:j1]))
        y[i] = 1.0 if h_future >= (1.0 + target_pct) * p0 else 0.0

    out = feat[["ts"] + FEATURE_COLUMNS].copy()
    out["y"] = y
    out = out.dropna(subset=["y"])
    return out


async def train_model(
    settings: Settings,
    model_dir: Path,
    coinbase_client,
    universe: List[str],
    status_store: JsonStore,
) -> ModelBundle:
    """
    Runs training and persists model bundle under MODEL_DIR/model/bundle.joblib.
    Updates status_store throughout.
    """
    store = CandleStore(model_dir / "cache" / "candles")
    end_dt = floor_to_5m(datetime.now(timezone.utc))

    started = utc_now_iso()
    status_store.write({
        "running": True,
        "started_at_utc": started,
        "finished_at_utc": None,
        "stage": "fetch_benchmark",
        "progress": {"done": 0, "total": len(universe)},
        "last_error": None,
        "metrics": None,
    })

    # Benchmark candles
    bench_df = await fetch_candles_full_lookback(
        coinbase_client,
        store,
        settings.BENCHMARK_SYMBOL,
        end_dt,
        settings.TRAIN_LOOKBACK_DAYS,
    )
    bench_reg = None
    if bench_df is not None and not bench_df.empty:
        bench_reg = compute_benchmark_regime(bench_df)

    rows = []
    total = len(universe)

    def progress_cb(info: Dict[str, Any]):
        # throttle writes by only writing when chunk changes (caller decides)
        pass

    for idx, pid in enumerate(universe, start=1):
        status_store.write({
            "running": True,
            "started_at_utc": started,
            "finished_at_utc": None,
            "stage": "fetch_and_build",
            "current_product": pid,
            "progress": {"done": idx - 1, "total": total},
            "last_error": None,
            "metrics": None,
        })

        df = await fetch_candles_full_lookback(
            coinbase_client,
            store,
            pid,
            end_dt,
            settings.TRAIN_LOOKBACK_DAYS,
        )
        if df is None or df.empty or len(df) < settings.MIN_BARS_5M:
            continue

        part = build_labels_and_features_for_product(
            df=df,
            bench_regime=bench_reg,
            target_pct=settings.TARGET_PCT,
            horizon_minutes=settings.HORIZON_MINUTES,
            decision_every_n_5m=settings.DECISION_EVERY_N_5M,
        )
        part["product"] = pid
        rows.append(part)

        # Hard cap to avoid blowing memory
        if sum(len(x) for x in rows) > settings.MAX_ROWS:
            break

    if not rows:
        raise RuntimeError("No training rows produced (insufficient candles or too strict settings).")

    data = pd.concat(rows, ignore_index=True).sort_values("ts")

    # Enforce MAX_ROWS: keep most recent rows
    if len(data) > settings.MAX_ROWS:
        data = data.iloc[-settings.MAX_ROWS:].copy()

    status_store.write({
        "running": True,
        "started_at_utc": started,
        "finished_at_utc": None,
        "stage": "fit_model",
        "progress": {"done": total, "total": total},
        "rows": int(len(data)),
        "last_error": None,
        "metrics": None,
    })

    bundle, grid_df = train_logreg_elasticnet_calibrated(
        data[["ts"] + FEATURE_COLUMNS + ["y"]],
        feature_cols=FEATURE_COLUMNS,
        c_values=settings.enet_c_values,
        l1_values=settings.enet_l1_values,
    )

    # fill config metadata
    bundle.target_pct = settings.TARGET_PCT
    bundle.horizon_minutes = settings.HORIZON_MINUTES
    bundle.benchmark_symbol = settings.BENCHMARK_SYMBOL
    bundle.decision_every_n_5m = settings.DECISION_EVERY_N_5M

    save_bundle(model_dir, bundle)

    # Persist training artifacts
    (model_dir / "model" / "grid_search.csv").parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(model_dir / "model" / "grid_search.csv", index=False)
    data[["ts", "y", "product"]].to_csv(model_dir / "model" / "training_index.csv", index=False)

    finished = utc_now_iso()
    status_store.write({
        "running": False,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "stage": "done",
        "progress": {"done": total, "total": total},
        "rows": int(len(data)),
        "last_error": None,
        "metrics": bundle.metrics,
        "bundle": bundle.to_dict(),
    })

    return bundle
