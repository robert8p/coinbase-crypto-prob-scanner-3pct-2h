from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


CANDLE_COLUMNS = ["ts", "low", "high", "open", "close", "volume"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CandleStore:
    base_dir: Path  # MODEL_DIR/cache/candles

    def _path(self, product_id: str) -> Path:
        safe = product_id.replace("/", "_")
        return self.base_dir / f"{safe}.csv.gz"

    def load(self, product_id: str) -> Optional[pd.DataFrame]:
        p = self._path(product_id)
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, compression="gzip")
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def save(self, product_id: str, df: pd.DataFrame) -> None:
        p = self._path(product_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        df.to_csv(p, index=False, compression="gzip")

    def upsert_rows(self, product_id: str, rows: pd.DataFrame) -> pd.DataFrame:
        existing = self.load(product_id)
        if existing is None or existing.empty:
            df = rows.copy()
        else:
            df = pd.concat([existing, rows], ignore_index=True)
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        self.save(product_id, df)
        return df

    def last_ts(self, product_id: str) -> Optional[int]:
        df = self.load(product_id)
        if df is None or df.empty:
            return None
        return int(df["ts"].max())


@dataclass
class JsonStore:
    path: Path

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def write(self, obj: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
