from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cache import JsonStore
from .config import Settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_product_id(pid: str) -> Tuple[str, str]:
    # Coinbase uses BASE-QUOTE
    if "-" in pid:
        base, quote = pid.split("-", 1)
        return base.upper(), quote.upper()
    return pid.upper(), ""


@dataclass
class UniverseResult:
    products: List[str]
    source: str
    warning: Optional[str]
    refreshed_utc: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "products": self.products,
            "count": len(self.products),
            "source": self.source,
            "warning": self.warning,
            "refreshed_utc": self.refreshed_utc,
        }


class UniverseManager:
    def __init__(self, settings: Settings, model_dir: Path):
        self.s = settings
        self.model_dir = model_dir
        self.store = JsonStore(model_dir / "universe_cache.json")
        self.fallback_path = Path(__file__).resolve().parent / "data" / "fallback_universe.json"

    def _fallback(self, warning: str) -> UniverseResult:
        import json

        products: List[str] = []
        try:
            products = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except Exception:
            products = []

        # filter by quote allowlist if possible
        allow = set(self.s.quote_allowlist_list)
        products = [p for p in products if _parse_product_id(p)[1] in allow]
        return UniverseResult(
            products=products[: self.s.UNIVERSE_MAX],
            source="fallback",
            warning=warning,
            refreshed_utc=utc_now().isoformat(),
        )

    def load_cached(self) -> Optional[UniverseResult]:
        obj = self.store.read()
        if not obj:
            return None
        try:
            products = obj.get("products", [])
            refreshed_utc = obj.get("refreshed_utc", "")
            source = obj.get("source", "cache")
            warning = obj.get("warning")
            # freshness check
            if refreshed_utc:
                t = datetime.fromisoformat(refreshed_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
                if utc_now() - t > timedelta(minutes=self.s.UNIVERSE_REFRESH_MINUTES):
                    return None
            return UniverseResult(products=products, source=source, warning=warning, refreshed_utc=refreshed_utc)
        except Exception:
            return None

    def write_cached(self, result: UniverseResult) -> None:
        self.store.write(result.as_dict())

    async def refresh(self, coinbase_client) -> UniverseResult:
        # explicit universe override
        if self.s.explicit_universe_list:
            products = [p.strip() for p in self.s.explicit_universe_list if p.strip()]
            # enforce allowlist
            allow = set(self.s.quote_allowlist_list)
            products = [p for p in products if _parse_product_id(p)[1] in allow]
            res = UniverseResult(products=products[: self.s.UNIVERSE_MAX], source="env:CRYPTO_UNIVERSE", warning=None, refreshed_utc=utc_now().isoformat())
            self.write_cached(res)
            return res

        allow_quotes = set(self.s.quote_allowlist_list)
        stable_bases = set(self.s.stablecoin_bases_list)

        try:
            prods = await coinbase_client.list_products()
        except Exception as e:
            return self._fallback(warning=f"/products failed: {type(e).__name__}: {e}")

        # filter
        filtered = []
        for p in prods:
            if p.get("status") not in (None, "", "online"):
                continue
            pid = p.get("id")
            if not pid:
                continue
            base = (p.get("base_currency") or "").upper()
            quote = (p.get("quote_currency") or "").upper()
            if quote not in allow_quotes:
                continue
            if self.s.EXCLUDE_STABLECOIN_BASE and base in stable_bases:
                continue
            filtered.append(pid)

        if not filtered:
            return self._fallback(warning="No products after filtering; using fallback list.")

        # Rank by approx 24h dollar volume (volume * last)
        async def score(pid: str):
            try:
                st = await coinbase_client.product_stats(pid)
                last = float(st.get("last") or 0.0)
                vol = float(st.get("volume") or 0.0)
                dv = last * vol
                return pid, dv
            except Exception:
                return pid, 0.0

        # Concurrency handled by coinbase_client itself; still gather
        scored = await asyncio.gather(*[score(pid) for pid in filtered])
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [pid for pid, dv in scored if dv > 0][: self.s.UNIVERSE_MAX]
        if not top:
            top = [pid for pid, dv in scored][: self.s.UNIVERSE_MAX]

        res = UniverseResult(products=top, source="coinbase:/products+stats", warning=None, refreshed_utc=utc_now().isoformat())
        self.write_cached(res)
        return res
