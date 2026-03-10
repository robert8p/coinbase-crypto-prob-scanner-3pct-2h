# Coinbase Crypto Prob Scanner (3% in 2h)

A production-ready FastAPI web app that scans a configurable universe of **Coinbase Exchange** spot products and shows:

- **Prob 3%**: probability the asset reaches at least **+3%** from scan time within a **2-hour (120 min)** forward horizon.

**Market data source:** Coinbase Exchange REST only (no trade execution).

**Scan cadence:** every **30 minutes**, aligned to **:00 and :30 UTC**.

---

## What “Prob 3%” means (exact definition)

Decision time is aligned to the **5-minute bar close**.

For each product at scan time:

- `P0` = current price at scan time (latest 5m close aligned to the scan)
- `H_future` = maximum future **5-minute HIGH** from scan time until `(scan time + HORIZON_MINUTES)`
- `Y=1` if `H_future >= (1 + TARGET_PCT) * P0` else `0`

The UI shows a **true probability 0–1** (not a rank).

---

## Model strategy

### Day-1 behavior
If no trained model exists, the app uses a **deterministic heuristic** to produce safe fallback probabilities.  
`/api/status` clearly indicates **heuristic** vs **trained**.

### Training
- Target: **+3% in 120 minutes**
- Base model: **Logistic Regression (elastic-net)** (`solver=saga`)
- Time split: contiguous **Train → Validation → Test** (most recent segment is test)
- Calibration: chooses **Platt (sigmoid)** vs **isotonic** by **validation Brier**
- Metrics reported:
  - AUC, PR AUC, Brier, ECE
  - event_rate (all vs test)

**Feature schema is fixed** and enforced at runtime. If schema mismatches, the app falls back to heuristic and warns.

---

## Feature schema (fixed)

Computed on **5-minute candles** and used identically in training and runtime scoring.

Momentum:
- ret_5m, ret_15m, ret_30m, ret_60m, impulse_60m, accel_30_60

Volatility:
- atr_pct, rv_60m, rv_6h, range_pct, range_vs_med, vol_of_vol

Breakout:
- donch_dist_6h, donch_dist_24h, bb_bw, bb_bw_pct

BTC regime:
- btc_ret_30m, btc_rv_6h (benchmark `BTC-USD` by default, override via `BENCHMARK_SYMBOL`)

Time:
- tod_sin, tod_cos, dow_sin, dow_cos

**Dropped (non-negotiable):**
- log_dvol, rvol_6h, obv_slope_6h

---

## Universe selection (liquidity-aware)

Default:
- Include products whose quote is in `QUOTE_ALLOWLIST` (default `USD,USDT,USDC`)
- Exclude stablecoin-base pairs by default (base in `STABLECOIN_BASES`)
- Compute approx 24h dollar volume from `/products/{id}/stats`:
  - `dollar_volume ≈ volume * last`
- Select top `UNIVERSE_MAX` for scanning (default 250)
- Use top `TRAIN_UNIVERSE_MAX` for training (default 100)
- If Coinbase `/products` fails, uses a baked-in fallback list in `app/data/fallback_universe.json`

Override universe explicitly:
- `CRYPTO_UNIVERSE="BTC-USD,ETH-USD,..."`

---

## Endpoints

- `GET /` : dashboard
- `GET /health` : 200 OK health check
- `GET /api/status` : coinbase/model/training/coverage + config
- `GET /api/scores` : latest rows with `prob_3`
- `POST /train` : start training (**password protected**)
- `GET /api/training/status` : training progress + metrics
- `GET /api/debug/coverage?password=...` : coverage + skipped products (max 200)

---

## Local run (optional)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ADMIN_PASSWORD="change-me"  # Windows: set ADMIN_PASSWORD=change-me
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Open: `http://localhost:8000`

---

## Deploy on Render (Step-by-Step)

1) **Create a new GitHub repo** and push this code (no edits needed).

2) In Render:
- **New → Web Service**
- Connect your GitHub repo
- Environment: **Docker**
- Branch: your default branch
- Health Check Path: `/health`

3) Add a **Persistent Disk**:
- Mount path: `/var/data`
- Size: 1–5 GB (start small; increase if you train large universes)

4) Set environment variables:
- `MODEL_DIR=/var/data/model`
- `ADMIN_PASSWORD=...` (required to enable training)

5) Deploy.

---

## Environment variables

### Core
- `TARGET_PCT` (default `0.03`)
- `HORIZON_MINUTES` (default `120`)
- `SCAN_INTERVAL_MINUTES` (default `30`)
- `TIMEZONE` (default `UTC`)
- `BENCHMARK_SYMBOL` (default `BTC-USD`)

### Universe
- `CRYPTO_UNIVERSE` (optional explicit list)
- `QUOTE_ALLOWLIST` (default `USD,USDT,USDC`)
- `EXCLUDE_STABLECOIN_BASE` (default `true`)
- `STABLECOIN_BASES` (default `USDT,USDC,DAI,TUSD,USDP,GUSD,FRAX,LUSD,EURC,BUSD`)
- `UNIVERSE_MAX` (default `250`)
- `TRAIN_UNIVERSE_MAX` (default `100`)
- `UNIVERSE_REFRESH_MINUTES` (default `360`)

### Rate limiting
- `COINBASE_BASE_URL` (default `https://api.exchange.coinbase.com`)
- `COINBASE_MAX_RPS` (default `5`)
- `COINBASE_MAX_INFLIGHT` (default `5`)
- `CANDLE_FETCH_BATCH` (default `6`)

### Data quality
- `MIN_BARS_5M` (default `300`)
- `MAX_CANDLE_STALENESS_MINUTES` (default `60`)

### Training
- `ADMIN_PASSWORD` (**required**)
- `TRAIN_LOOKBACK_DAYS` (default `180`)
- `DECISION_EVERY_N_5M` (default `3`)
- `MAX_ROWS` (default `2000000`)
- `ENET_C_VALUES` (default `0.5,1.0`)
- `ENET_L1_VALUES` (default `0.0,0.5`)

### Storage
- `MODEL_DIR` (default `./runtime/model`, **Render recommended** `/var/data/model`)

### Debug
- `DEMO_MODE` (default `false`)
- `DISABLE_SCHEDULER` (default `0`)
- `DEBUG_PASSWORD` (optional; if unset, debug endpoint uses `ADMIN_PASSWORD`)

---

## Copy/paste env bundles

### Debug bundle (safe, no external calls)
```
DEMO_MODE=true
DISABLE_SCHEDULER=1
ADMIN_PASSWORD=change-me
MODEL_DIR=/var/data/model
TARGET_PCT=0.03
HORIZON_MINUTES=120
SCAN_INTERVAL_MINUTES=30
QUOTE_ALLOWLIST=USD,USDT,USDC
UNIVERSE_MAX=250
TRAIN_UNIVERSE_MAX=100
```

### Live bundle
```
DEMO_MODE=false
DISABLE_SCHEDULER=0
ADMIN_PASSWORD=change-me
MODEL_DIR=/var/data/model
TARGET_PCT=0.03
HORIZON_MINUTES=120
SCAN_INTERVAL_MINUTES=30
COINBASE_MAX_RPS=5
COINBASE_MAX_INFLIGHT=5
CANDLE_FETCH_BATCH=6
QUOTE_ALLOWLIST=USD,USDT,USDC
UNIVERSE_MAX=250
TRAIN_UNIVERSE_MAX=100
```

---

## Go-live checklist

- [ ] Persistent disk mounted at `/var/data`
- [ ] `MODEL_DIR=/var/data/model`
- [ ] `ADMIN_PASSWORD` set
- [ ] `DEMO_MODE=false`
- [ ] Confirm `/health` returns 200
- [ ] Confirm `/api/status` shows recent Coinbase OK time after a scan
- [ ] Run training once from the UI and confirm `/api/status` shows **trained**

---

## Notes on performance / limits

- Coinbase candles endpoint returns max ~300 candles per request; initial training with large lookbacks can be request-heavy.
- The app uses disk-cached candles and incremental refresh, so repeated runs typically fetch only recent data.
- Scheduler is protected by a **file lock** in `MODEL_DIR` and the Docker start command enforces a **single worker**.

---

## License

MIT (add your preferred license if needed).
