from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


class Settings(BaseSettings):
    """
    Env-var only configuration (no code edits required).

    All defaults are production-safe. For Render, mount a disk at /var/data and set:
      MODEL_DIR=/var/data/model
    """
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False)

    # Core
    TARGET_PCT: float = Field(default=0.03)
    HORIZON_MINUTES: int = Field(default=120)
    SCAN_INTERVAL_MINUTES: int = Field(default=30)
    TIMEZONE: str = Field(default="UTC")
    BENCHMARK_SYMBOL: str = Field(default="BTC-USD")

    # Universe
    CRYPTO_UNIVERSE: Optional[str] = Field(default=None)  # explicit list, comma-separated
    QUOTE_ALLOWLIST: str = Field(default="USD,USDT,USDC")
    EXCLUDE_STABLECOIN_BASE: bool = Field(default=True)
    STABLECOIN_BASES: str = Field(default="USDT,USDC,DAI,TUSD,USDP,GUSD,FRAX,LUSD,EURC,BUSD")
    UNIVERSE_MAX: int = Field(default=250)
    TRAIN_UNIVERSE_MAX: int = Field(default=100)
    UNIVERSE_REFRESH_MINUTES: int = Field(default=360)

    # Rate limiting / requests
    COINBASE_BASE_URL: str = Field(default="https://api.exchange.coinbase.com")
    COINBASE_MAX_RPS: float = Field(default=5.0)
    COINBASE_MAX_INFLIGHT: int = Field(default=5)
    CANDLE_FETCH_BATCH: int = Field(default=6)  # number of per-product candle chunk requests to do per scan loop

    # Data quality
    MIN_BARS_5M: int = Field(default=300)
    MAX_CANDLE_STALENESS_MINUTES: int = Field(default=60)

    # Training
    ADMIN_PASSWORD: str = Field(default="")
    TRAIN_LOOKBACK_DAYS: int = Field(default=180)
    DECISION_EVERY_N_5M: int = Field(default=3)
    MAX_ROWS: int = Field(default=2_000_000)
    ENET_C_VALUES: str = Field(default="0.5,1.0")
    ENET_L1_VALUES: str = Field(default="0.0,0.5")

    # Storage
    MODEL_DIR: str = Field(default="./runtime/model")

    # Debug
    DEMO_MODE: bool = Field(default=False)
    DISABLE_SCHEDULER: int = Field(default=0)
    DEBUG_PASSWORD: Optional[str] = Field(default=None)

    # Internal
    USER_AGENT: str = Field(default="CoinbaseCryptoProbScanner/1.0")

    @property
    def quote_allowlist_list(self) -> List[str]:
        return _parse_csv(self.QUOTE_ALLOWLIST.upper())

    @property
    def stablecoin_bases_list(self) -> List[str]:
        return _parse_csv(self.STABLECOIN_BASES.upper())

    @property
    def explicit_universe_list(self) -> Optional[List[str]]:
        if not self.CRYPTO_UNIVERSE:
            return None
        return _parse_csv(self.CRYPTO_UNIVERSE)

    @property
    def enet_c_values(self) -> List[float]:
        return [float(x) for x in _parse_csv(self.ENET_C_VALUES)]

    @property
    def enet_l1_values(self) -> List[float]:
        return [float(x) for x in _parse_csv(self.ENET_L1_VALUES)]

    def ensure_dirs(self) -> Path:
        d = Path(self.MODEL_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / "cache" / "candles").mkdir(parents=True, exist_ok=True)
        (d / "model").mkdir(parents=True, exist_ok=True)
        return d


def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
