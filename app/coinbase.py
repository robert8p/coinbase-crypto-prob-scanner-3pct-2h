from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import Settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RateLimitStats:
    requests: int = 0
    ok: int = 0
    http_429: int = 0
    http_5xx: int = 0
    other_errors: int = 0
    retries: int = 0
    backoff_seconds_total: float = 0.0
    last_error: Optional[str] = None
    last_ok_utc: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "ok": self.ok,
            "http_429": self.http_429,
            "http_5xx": self.http_5xx,
            "other_errors": self.other_errors,
            "retries": self.retries,
            "backoff_seconds_total": round(self.backoff_seconds_total, 3),
            "last_error": self.last_error,
            "last_ok_utc": self.last_ok_utc,
        }


class AsyncTokenBucket:
    """
    Simple rate limiter: ensures average rate <= max_rps.
    Not a perfect token bucket, but reliable and deterministic under async concurrency.
    """
    def __init__(self, max_rps: float):
        self.max_rps = max(0.1, float(max_rps))
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            if wait > 0:
                await asyncio.sleep(wait)
            # schedule next slot
            self._next_allowed = max(self._next_allowed, time.monotonic()) + (1.0 / self.max_rps)


class CoinbaseClient:
    def __init__(self, settings: Settings, stats: RateLimitStats):
        self.s = settings
        self.stats = stats
        self._bucket = AsyncTokenBucket(settings.COINBASE_MAX_RPS)
        self._sem = asyncio.Semaphore(settings.COINBASE_MAX_INFLIGHT)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "CoinbaseClient":
        headers = {"User-Agent": self.s.USER_AGENT, "Accept": "application/json"}
        self._client = httpx.AsyncClient(base_url=self.s.COINBASE_BASE_URL, headers=headers, timeout=httpx.Timeout(20.0))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
        self._client = None

    async def _request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        assert self._client is not None, "Client not started"
        max_tries = 6
        base_backoff = 0.6

        for attempt in range(max_tries):
            await self._bucket.acquire()
            async with self._sem:
                self.stats.requests += 1
                try:
                    r = await self._client.request(method, url, params=params)
                except Exception as e:
                    self.stats.other_errors += 1
                    self.stats.last_error = f"{type(e).__name__}: {e}"
                    if attempt < max_tries - 1:
                        self.stats.retries += 1
                        backoff = base_backoff * (2 ** attempt) * (0.5 + random.random())
                        self.stats.backoff_seconds_total += backoff
                        await asyncio.sleep(backoff)
                        continue
                    raise

            if r.status_code == 200:
                self.stats.ok += 1
                self.stats.last_ok_utc = utc_now().isoformat()
                return r.json()

            # Retry-safe status codes
            if r.status_code == 429 or 500 <= r.status_code <= 599:
                if r.status_code == 429:
                    self.stats.http_429 += 1
                else:
                    self.stats.http_5xx += 1

                self.stats.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                if attempt < max_tries - 1:
                    self.stats.retries += 1
                    # respect Retry-After if present
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        try:
                            backoff = float(retry_after)
                        except Exception:
                            backoff = base_backoff * (2 ** attempt)
                    else:
                        backoff = base_backoff * (2 ** attempt)
                    backoff = backoff * (0.7 + random.random() * 0.6)  # jitter
                    self.stats.backoff_seconds_total += backoff
                    await asyncio.sleep(backoff)
                    continue

            # Non-retry errors
            self.stats.other_errors += 1
            self.stats.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            r.raise_for_status()

        raise RuntimeError("Request failed after retries")

    async def list_products(self) -> List[Dict[str, Any]]:
        return await self._request("GET", "/products")

    async def product_stats(self, product_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/products/{product_id}/stats")

    async def candles(self, product_id: str, start_iso: str, end_iso: str, granularity: int = 300) -> List[List[Any]]:
        params = {"start": start_iso, "end": end_iso, "granularity": granularity}
        return await self._request("GET", f"/products/{product_id}/candles", params=params)


class DemoCoinbaseClient:
    """
    Demo mode: deterministic synthetic data; no external calls.
    Produces plausible candles for a small baked-in universe.
    """
    def __init__(self, settings: Settings, stats: RateLimitStats):
        self.s = settings
        self.stats = stats
        self._rng = random.Random(42)

    async def __aenter__(self) -> "DemoCoinbaseClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def list_products(self) -> List[Dict[str, Any]]:
        self.stats.ok += 1
        self.stats.requests += 1
        # mimic Coinbase product schema (subset)
        products = []
        for pid in ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "ADA-USD"]:
            base, quote = pid.split("-")
            products.append({"id": pid, "base_currency": base, "quote_currency": quote, "status": "online"})
        return products

    async def product_stats(self, product_id: str) -> Dict[str, Any]:
        self.stats.ok += 1
        self.stats.requests += 1
        last = 10.0 + self._rng.random() * 100.0
        vol = 10_000 + self._rng.random() * 500_000
        return {"last": str(last), "volume": str(vol)}

    async def candles(self, product_id: str, start_iso: str, end_iso: str, granularity: int = 300) -> List[List[Any]]:
        self.stats.ok += 1
        self.stats.requests += 1
        # Create synthetic 5m candles between start and end (max 300 points typical; respect caller chunking)
        start = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        step = granularity
        n = int((end - start).total_seconds() // step)
        n = max(0, min(n, 300))
        # seed per product for stable series
        seed = abs(hash(product_id)) % (2**31 - 1)
        rng = random.Random(seed + int(start.timestamp()))
        price = 20.0 + (seed % 1000) / 10.0
        rows = []
        ts = int(end.timestamp()) - n * step
        for i in range(n):
            # random walk with occasional spikes
            drift = rng.normalvariate(0, 0.0015)
            shock = 0.0
            if rng.random() < 0.02:
                shock = rng.normalvariate(0.01, 0.01)
            ret = drift + shock
            o = price
            c = price * (1 + ret)
            hi = max(o, c) * (1 + abs(rng.normalvariate(0, 0.001)))
            lo = min(o, c) * (1 - abs(rng.normalvariate(0, 0.001)))
            vol = abs(rng.normalvariate(100, 40))
            rows.append([ts, float(lo), float(hi), float(o), float(c), float(vol)])
            price = c
            ts += step
        # Coinbase returns newest-first, but we return newest-first to match real API
        rows.reverse()
        return rows
