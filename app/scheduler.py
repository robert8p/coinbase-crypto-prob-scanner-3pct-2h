from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from filelock import FileLock, Timeout

from .cache import CandleStore, JsonStore
from .coinbase import CoinbaseClient, DemoCoinbaseClient, RateLimitStats
from .config import Settings
from .features import FEATURE_COLUMNS, compute_benchmark_regime, compute_features, latest_feature_row
from .heuristics import heuristic_prob_and_notes
from .modeling import ModelBundle, load_bundle, predict_prob


GRANULARITY_SEC = 300  # 5m candles


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def floor_to_5m(dt: datetime) -> datetime:
    ts = int(dt.timestamp())
    ts -= ts % GRANULARITY_SEC
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def next_aligned_run(now: datetime, interval_min: int) -> datetime:
    # align to multiples of interval on epoch minutes
    epoch_min = int(now.timestamp() // 60)
    next_min = ((epoch_min // interval_min) + 1) * interval_min
    return datetime.fromtimestamp(next_min * 60, tz=timezone.utc)


def _candles_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "low", "high", "open", "close", "volume"])
    # Coinbase returns newest first, each: [time, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df = df.sort_values("ts")
    return df


async def fetch_candles_incremental(
    client,
    store: CandleStore,
    product_id: str,
    end_dt: datetime,
    lookback_days: int,
    batch_limit: int,
) -> Optional[pd.DataFrame]:
    """
    Disk-cached, incremental candle refresh.
    Fetch only missing segments up to batch_limit requests per call.
    """
    end_dt = floor_to_5m(end_dt)
    end_iso = iso_z(end_dt)
    existing = store.load(product_id)
    # Determine start time
    if existing is None or existing.empty:
        start_dt = end_dt - timedelta(days=lookback_days)
        start_dt = floor_to_5m(start_dt)
        start_ts = int(start_dt.timestamp())
        last_ts = None
    else:
        last_ts = int(existing["ts"].max())
        # fetch from just after last known candle
        start_ts = last_ts + GRANULARITY_SEC

    if start_ts >= int(end_dt.timestamp()):
        return existing

    # Coinbase max 300 candles per request => span = 300*5m = 25h
    max_span_sec = GRANULARITY_SEC * 300

    # We'll fetch chunks starting from start_ts to end, but capped by batch_limit.
    chunks = []
    cur = start_ts
    end_ts = int(end_dt.timestamp())
    while cur < end_ts and len(chunks) < batch_limit:
        chunk_end = min(cur + max_span_sec, end_ts)
        chunks.append((cur, chunk_end))
        cur = chunk_end

    # If we have more remaining than batch_limit, we prioritize the most recent chunks:
    if cur < end_ts:
        # rebuild chunks for last batch_limit windows ending at end_ts
        chunks = []
        cur_end = end_ts
        for _ in range(batch_limit):
            cur_start = max(start_ts, cur_end - max_span_sec)
            chunks.append((cur_start, cur_end))
            cur_end = cur_start
            if cur_end <= start_ts:
                break
        chunks.reverse()

    all_new = []
    for s_ts, e_ts in chunks:
        if e_ts <= s_ts:
            continue
        s_iso = iso_z(datetime.fromtimestamp(s_ts, tz=timezone.utc))
        e_iso = iso_z(datetime.fromtimestamp(e_ts, tz=timezone.utc))
        rows = await client.candles(product_id, s_iso, e_iso, granularity=GRANULARITY_SEC)
        df_new = _candles_to_df(rows)
        if not df_new.empty:
            all_new.append(df_new)

    if all_new:
        new_df = pd.concat(all_new, ignore_index=True)
        merged = store.upsert_rows(product_id, new_df)
        return merged
    return existing


@dataclass
class Coverage:
    universe_count: int = 0
    symbols_requested_count: int = 0
    symbols_returned_with_candles_count: int = 0
    symbols_with_sufficient_bars_count: int = 0
    scored_count: int = 0
    top_skip_reasons: Dict[str, int] = field(default_factory=dict)
    last_run_utc: Optional[str] = None
    last_candle_timestamp: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "universe_count": self.universe_count,
            "requested": self.symbols_requested_count,
            "returned_with_candles": self.symbols_returned_with_candles_count,
            "sufficient_bars": self.symbols_with_sufficient_bars_count,
            "scored_count": self.scored_count,
            "top_skip_reasons": self.top_skip_reasons,
            "last_run_utc": self.last_run_utc,
            "last_candle_timestamp": self.last_candle_timestamp,
        }


@dataclass
class AppState:
    settings: Settings
    model_dir: Path
    rate_stats: RateLimitStats = field(default_factory=RateLimitStats)
    coverage: Coverage = field(default_factory=Coverage)
    scores: List[Dict[str, Any]] = field(default_factory=list)
    debug_skips: List[Dict[str, Any]] = field(default_factory=list)
    bundle: Optional[ModelBundle] = None
    model_status: str = "heuristic"  # or "trained"
    model_warning: Optional[str] = None
    training_status_store: JsonStore = field(init=False)
    training_running: bool = False

    def __post_init__(self):
        self.training_status_store = JsonStore(self.model_dir / "training_status.json")


def load_model_if_any(state: AppState) -> None:
    state.bundle = load_bundle(state.model_dir)
    state.model_warning = None
    if state.bundle is None:
        state.model_status = "heuristic"
        return

    # Schema check: must match features and config (target/horizon)
    b = state.bundle
    if b.feature_cols != FEATURE_COLUMNS:
        state.model_status = "heuristic"
        state.model_warning = "Model schema mismatch; retrain required."
        state.bundle = None
        return

    # Config mismatch isn't fatal but warn
    if abs(b.target_pct - state.settings.TARGET_PCT) > 1e-9 or int(b.horizon_minutes) != int(state.settings.HORIZON_MINUTES):
        state.model_warning = f"Model trained for target={b.target_pct:.3f}, horizon={b.horizon_minutes}m; current env target={state.settings.TARGET_PCT:.3f}, horizon={state.settings.HORIZON_MINUTES}m."

    state.model_status = "trained"


def _candle_ts_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _is_stale(last_ts: int, max_stale_min: int) -> bool:
    # candle 'ts' is candle start; close is start+5m. We'll use close time.
    close_time = datetime.fromtimestamp(last_ts + GRANULARITY_SEC, tz=timezone.utc)
    age_min = (utc_now() - close_time).total_seconds() / 60.0
    return age_min > max_stale_min


async def scan_once(state: AppState) -> None:
    s = state.settings
    model_dir = state.model_dir
    store = CandleStore(model_dir / "cache" / "candles")

    from .universe import UniverseManager
    um = UniverseManager(s, model_dir)

    skip_reasons: Dict[str, int] = {}

    def skip(pid: str, reason: str):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if len(state.debug_skips) < 200:
            state.debug_skips.append({"product": pid, "reason": reason})

    state.coverage = Coverage()
    state.debug_skips = []
    state.coverage.last_run_utc = utc_now().isoformat().replace("+00:00", "Z")

    end_dt = floor_to_5m(utc_now())

    ClientCls = DemoCoinbaseClient if s.DEMO_MODE else CoinbaseClient

    async with ClientCls(s, state.rate_stats) as cb:
        # universe
        cached = um.load_cached()
        if cached is None:
            u = await um.refresh(cb)
        else:
            u = cached
        universe = u.products
        state.coverage.universe_count = len(universe)
        state.coverage.symbols_requested_count = len(universe)

        # benchmark candles
        bench_df = await fetch_candles_incremental(
            cb, store, s.BENCHMARK_SYMBOL, end_dt, lookback_days=max(10, s.TRAIN_LOOKBACK_DAYS), batch_limit=s.CANDLE_FETCH_BATCH
        )
        bench_reg = None
        if bench_df is not None and not bench_df.empty and len(bench_df) >= s.MIN_BARS_5M:
            bench_reg = compute_benchmark_regime(bench_df)
        else:
            skip(s.BENCHMARK_SYMBOL, "benchmark_missing")

        results: List[Dict[str, Any]] = []
        last_candle_ts_seen: Optional[int] = None

        for pid in universe:
            try:
                df = await fetch_candles_incremental(
                    cb, store, pid, end_dt, lookback_days=10, batch_limit=s.CANDLE_FETCH_BATCH
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if "429" in msg:
                    skip(pid, "rate_limited")
                else:
                    skip(pid, "other_errors")
                continue

            if df is None or df.empty:
                skip(pid, "no_candles")
                continue
            state.coverage.symbols_returned_with_candles_count += 1

            if len(df) < s.MIN_BARS_5M:
                skip(pid, "insufficient_candles")
                continue
            state.coverage.symbols_with_sufficient_bars_count += 1

            last_ts = int(df["ts"].max())
            if last_candle_ts_seen is None or last_ts > last_candle_ts_seen:
                last_candle_ts_seen = last_ts

            stale = _is_stale(last_ts, s.MAX_CANDLE_STALENESS_MINUTES)
            if stale:
                skip(pid, "stale_candles")
                # still compute, but flag

            feat = compute_features(df, bench_regime=bench_reg)
            row = latest_feature_row(feat)
            if row is None:
                skip(pid, "feature_error")
                continue

            price = float(row["close"])
            last_candle_time = _candle_ts_to_iso(int(row["ts"]) + GRANULARITY_SEC)

            # model vs heuristic
            prob_source = "heuristic"
            notes = ""
            risk = "OK"
            prob = None
            if state.bundle is not None:
                try:
                    X = feat.iloc[[-1]][FEATURE_COLUMNS].copy()
                    if X.isna().any(axis=1).iloc[0]:
                        raise ValueError("NaNs in features")
                    prob = float(predict_prob(state.bundle, X)[0])
                    prob_source = "model"
                    notes = "model"
                    # simple risk: use rv_6h
                    rv6h = float(row.get("rv_6h") or 0.0)
                    if rv6h > 0.02:
                        risk = "CAUTION"
                    if rv6h > 0.035:
                        risk = "HIGH"
                except Exception:
                    prob, notes, risk = heuristic_prob_and_notes(row)
                    prob_source = "heuristic"
            else:
                prob, notes, risk = heuristic_prob_and_notes(row)

            if stale:
                notes = (notes + ";STALE_CANDLES").strip(";")

            results.append({
                "product": pid,
                "price": price,
                "prob_3": float(prob),
                "prob_3_source": prob_source,
                "quote": pid.split("-")[-1] if "-" in pid else "",
                "last_candle_time": last_candle_time,
                "notes": notes,
                "risk": risk,
            })

        # sort by prob desc
        results.sort(key=lambda x: x.get("prob_3", 0.0), reverse=True)
        state.scores = results
        state.coverage.scored_count = len(results)
        state.coverage.top_skip_reasons = dict(sorted(skip_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10])

        if last_candle_ts_seen is not None:
            state.coverage.last_candle_timestamp = _candle_ts_to_iso(last_candle_ts_seen + GRANULARITY_SEC)


async def scheduler_loop(state: AppState, stop_event: asyncio.Event) -> None:
    # initial short delay to allow app to start
    await asyncio.sleep(0.5)
    while not stop_event.is_set():
        try:
            await scan_once(state)
        except Exception as e:
            state.rate_stats.last_error = f"scan_once: {type(e).__name__}: {e}"

        # sleep until next aligned run
        now = utc_now()
        nxt = next_aligned_run(now, max(1, int(state.settings.SCAN_INTERVAL_MINUTES)))
        sleep_s = max(5.0, (nxt - now).total_seconds())
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_s)
        except asyncio.TimeoutError:
            pass


def try_acquire_scheduler_lock(model_dir: Path) -> Optional[FileLock]:
    lock_path = model_dir / "scheduler.lock"
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
        return lock
    except Timeout:
        return None


def try_acquire_training_lock(model_dir: Path) -> Optional[FileLock]:
    lock_path = model_dir / "training.lock"
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
        return lock
    except Timeout:
        return None
