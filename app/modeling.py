from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def schema_hash(feature_cols: List[str]) -> str:
    h = hashlib.sha256(",".join(feature_cols).encode("utf-8")).hexdigest()
    return h[:16]


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    # ECE: sum_k |acc_k - conf_k| * (n_k / n)
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = p[mask].mean()
        ece += abs(acc - conf) * (mask.sum() / n)
    return float(ece)


@dataclass
class ModelBundle:
    feature_cols: List[str]
    target_pct: float
    horizon_minutes: int
    benchmark_symbol: str
    decision_every_n_5m: int
    trained_utc: str
    schema_hash: str
    metrics: Dict[str, Any]
    calibrator_method: str
    model: Any  # CalibratedClassifierCV

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_cols": self.feature_cols,
            "target_pct": self.target_pct,
            "horizon_minutes": self.horizon_minutes,
            "benchmark_symbol": self.benchmark_symbol,
            "decision_every_n_5m": self.decision_every_n_5m,
            "trained_utc": self.trained_utc,
            "schema_hash": self.schema_hash,
            "metrics": self.metrics,
            "calibrator_method": self.calibrator_method,
        }


def bundle_path(model_dir: Path) -> Path:
    return model_dir / "model" / "bundle.joblib"


def save_bundle(model_dir: Path, bundle: ModelBundle) -> None:
    p = bundle_path(model_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)


def load_bundle(model_dir: Path) -> Optional[ModelBundle]:
    p = bundle_path(model_dir)
    if not p.exists():
        return None
    try:
        b = joblib.load(p)
        return b
    except Exception:
        return None


def _split_time(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def _fit_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    l1_ratio: float,
) -> Tuple[float, str, CalibratedClassifierCV]:
    """
    Returns (val_brier, method, calibrated_model)
    Picks best between sigmoid and isotonic based on validation Brier.
    """
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            C=float(C),
            l1_ratio=float(l1_ratio),
            max_iter=3000,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        )),
    ])

    base.fit(X_train, y_train)

    best = None
    best_method = None
    best_cal = None

    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(estimator=base, cv="prefit", method=method)
        cal.fit(X_val, y_val)
        p_val = cal.predict_proba(X_val)[:, 1]
        brier = float(brier_score_loss(y_val, p_val))
        if (best is None) or (brier < best):
            best = brier
            best_method = method
            best_cal = cal

    assert best_cal is not None
    return float(best), str(best_method), best_cal


def train_logreg_elasticnet_calibrated(
    df: pd.DataFrame,
    feature_cols: List[str],
    c_values: List[float],
    l1_values: List[float],
) -> Tuple[ModelBundle, pd.DataFrame]:
    """
    df must have columns: ts, y plus all features.
    Returns: (bundle, metrics_df)
    """
    df = df.dropna(subset=feature_cols + ["y"]).copy()
    df["y"] = df["y"].astype(int)

    train, val, test = _split_time(df)

    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=int)
    X_val = val[feature_cols].to_numpy(dtype=float)
    y_val = val["y"].to_numpy(dtype=int)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=int)

    candidates = []
    best_brier = None
    best = None

    for C in c_values:
        for l1 in l1_values:
            val_brier, method, cal = _fit_candidate(X_train, y_train, X_val, y_val, C, l1)

            # quick val diagnostics
            p_val = cal.predict_proba(X_val)[:, 1]
            auc_val = float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan")
            ap_val = float(average_precision_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else float("nan")

            row = {
                "C": float(C),
                "l1_ratio": float(l1),
                "calibration": method,
                "brier_val": val_brier,
                "auc_val": auc_val,
                "pr_auc_val": ap_val,
            }
            candidates.append(row)

            if (best_brier is None) or (val_brier < best_brier):
                best_brier = val_brier
                best = (C, l1, method, cal)

    assert best is not None
    C, l1, method, cal = best

    # Final metrics on test
    p_test = cal.predict_proba(X_test)[:, 1]
    metrics = {
        "auc_test": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "pr_auc_test": float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "brier_test": float(brier_score_loss(y_test, p_test)),
        "ece_test": expected_calibration_error(y_test, p_test, n_bins=15),
        "event_rate_all": float(df["y"].mean()),
        "event_rate_test": float(y_test.mean()) if len(y_test) else float("nan"),
        "rows_all": int(len(df)),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "best_C": float(C),
        "best_l1_ratio": float(l1),
        "best_calibration": method,
    }

    metrics_df = pd.DataFrame(candidates).sort_values("brier_val").reset_index(drop=True)

    # bundle filled later with config metadata
    dummy = ModelBundle(
        feature_cols=feature_cols,
        target_pct=0.03,
        horizon_minutes=120,
        benchmark_symbol="BTC-USD",
        decision_every_n_5m=3,
        trained_utc=utc_now_iso(),
        schema_hash=schema_hash(feature_cols),
        metrics=metrics,
        calibrator_method=method,
        model=cal,
    )
    return dummy, metrics_df


def predict_prob(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    return bundle.model.predict_proba(X[bundle.feature_cols].to_numpy(dtype=float))[:, 1]
