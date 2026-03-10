from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .scheduler import (
    AppState,
    load_model_if_any,
    scheduler_loop,
    try_acquire_scheduler_lock,
    try_acquire_training_lock,
)
from .training import train_model
from .coinbase import CoinbaseClient, DemoCoinbaseClient
from .universe import UniverseManager


app = FastAPI(title="Coinbase Crypto Prob Scanner (3% in 2h)", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

state: Optional[AppState] = None
_stop_event: Optional[asyncio.Event] = None
_scheduler_task: Optional[asyncio.Task] = None
_scheduler_lock = None  # file lock handle
_training_lock = None


@app.on_event("startup")
async def on_startup():
    global state, _stop_event, _scheduler_task, _scheduler_lock
    s = get_settings()
    model_dir = Path(s.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    state = AppState(settings=s, model_dir=model_dir)
    load_model_if_any(state)

    _stop_event = asyncio.Event()

    if int(s.DISABLE_SCHEDULER) == 1:
        return

    _scheduler_lock = try_acquire_scheduler_lock(model_dir)
    if _scheduler_lock is None:
        # Another worker already runs the scheduler
        if state:
            state.rate_stats.last_error = "Scheduler lock not acquired (another worker likely active)."
        return

    _scheduler_task = asyncio.create_task(scheduler_loop(state, _stop_event))


@app.on_event("shutdown")
async def on_shutdown():
    global _stop_event, _scheduler_task, _scheduler_lock
    try:
        if _stop_event:
            _stop_event.set()
        if _scheduler_task:
            await asyncio.wait_for(_scheduler_task, timeout=5.0)
    except Exception:
        pass
    finally:
        _scheduler_task = None
        _stop_event = None
        if _scheduler_lock:
            try:
                _scheduler_lock.release()
            except Exception:
                pass
            _scheduler_lock = None


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _require_state() -> AppState:
    if state is None:
        raise HTTPException(status_code=503, detail="App not ready")
    return state


@app.get("/api/scores")
def api_scores():
    st = _require_state()
    return {"rows": st.scores, "count": len(st.scores)}


@app.get("/api/training/status")
def api_training_status():
    st = _require_state()
    obj = st.training_status_store.read()
    if not obj:
        obj = {"running": False, "started_at_utc": None, "finished_at_utc": None, "stage": "idle", "last_error": None}
    return obj


@app.get("/api/status")
def api_status():
    st = _require_state()
    s = st.settings
    training = st.training_status_store.read() or {"running": False, "stage": "idle", "last_error": None}

    return {
        "demo_mode": bool(s.DEMO_MODE),
        "model_dir": str(st.model_dir),
        "config": {
            "target_pct": s.TARGET_PCT,
            "horizon_minutes": s.HORIZON_MINUTES,
            "scan_interval_minutes": s.SCAN_INTERVAL_MINUTES,
            "decision_every_n_5m": s.DECISION_EVERY_N_5M,
            "benchmark_symbol": s.BENCHMARK_SYMBOL,
            "quote_allowlist": s.quote_allowlist_list,
            "universe_max": s.UNIVERSE_MAX,
            "train_universe_max": s.TRAIN_UNIVERSE_MAX,
            "max_candle_staleness_minutes": s.MAX_CANDLE_STALENESS_MINUTES,
            "min_bars_5m": s.MIN_BARS_5M,
        },
        "coinbase": {
            "ok": st.rate_stats.last_ok_utc is not None,
            "last_ok_utc": st.rate_stats.last_ok_utc,
            "last_error": st.rate_stats.last_error,
            "rate_limit_stats": st.rate_stats.as_dict(),
            "base_url": s.COINBASE_BASE_URL,
            "max_rps": s.COINBASE_MAX_RPS,
            "max_inflight": s.COINBASE_MAX_INFLIGHT,
        },
        "model": {
            "status": st.model_status,
            "warning": st.model_warning,
            "bundle": st.bundle.to_dict() if st.bundle else None,
        },
        "coverage": st.coverage.as_dict(),
        "training": training,
    }


@app.post("/train")
async def train(payload: Dict[str, Any]):
    st = _require_state()
    s = st.settings

    if not s.ADMIN_PASSWORD:
        raise HTTPException(status_code=400, detail="ADMIN_PASSWORD env var is required to enable training.")
    if payload.get("password") != s.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    # Prevent concurrent training
    global _training_lock
    if _training_lock is not None:
        raise HTTPException(status_code=409, detail="Training already running")

    lock = try_acquire_training_lock(st.model_dir)
    if lock is None:
        raise HTTPException(status_code=409, detail="Training lock not acquired (another training run likely active).")
    _training_lock = lock

    # determine training universe
    um = UniverseManager(s, st.model_dir)
    ClientCls = DemoCoinbaseClient if s.DEMO_MODE else CoinbaseClient

    async def _run():
        global _training_lock
        try:
            async with ClientCls(s, st.rate_stats) as cb:
                cached = um.load_cached()
                if cached is None:
                    u = await um.refresh(cb)
                else:
                    u = cached
                train_universe = u.products[: max(1, int(s.TRAIN_UNIVERSE_MAX))]
                bundle = await train_model(s, st.model_dir, cb, train_universe, st.training_status_store)
                # reload model into runtime
                st.bundle = bundle
                st.model_status = "trained"
                st.model_warning = None
        except Exception as e:
            obj = st.training_status_store.read() or {}
            obj.update({"running": False, "stage": "error", "last_error": f"{type(e).__name__}: {e}"})
            st.training_status_store.write(obj)
        finally:
            try:
                if _training_lock:
                    _training_lock.release()
            except Exception:
                pass
            _training_lock = None

    asyncio.create_task(_run())
    return {"ok": True, "message": "Training started"}


@app.get("/api/debug/coverage")
def debug_coverage(password: str = ""):
    st = _require_state()
    s = st.settings
    expected = s.DEBUG_PASSWORD or s.ADMIN_PASSWORD
    if expected and password != expected:
        raise HTTPException(status_code=401, detail="Invalid password")

    # list skipped products is not tracked per product in this minimal version;
    # we provide aggregated skip reasons + current universe snapshot.
    um = UniverseManager(s, st.model_dir)
    cached = um.load_cached()
    return {
        "universe": cached.as_dict() if cached else None,
        "coverage": st.coverage.as_dict(),
        "skipped_products": st.debug_skips,
        "max_returned": 200,
    }
