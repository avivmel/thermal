"""
Drift model evaluation with per-home residual correction.

Trains and evaluates the passive drift time-to-boundary model using two setups:

1. Cross-home evaluation (train split -> test split homes):
   Tests true generalization to unseen homes. Per-home corrections cannot apply
   since test homes have no training history.

2. Within-home evaluation (70/30 within each training home):
   Shows the benefit of per-home residual correction for known homes.

Saves an updated artifact (models/drift_time_xgb.pkl) that includes per-home
residuals, compatible with the updated predict_drift_time() interface.

Usage:
    python scripts/evaluate_drift_model.py
    python scripts/evaluate_drift_model.py --no-save
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge

# GBM backend selection (mirrors run_xgboost_baselines.py)
GBM_BACKEND = None
try:
    import lightgbm as lgb
    GBM_BACKEND = "lightgbm"
except ImportError:
    try:
        import xgboost as xgb
        GBM_BACKEND = "xgboost"
    except (ImportError, Exception):
        pass

if GBM_BACKEND is None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    GBM_BACKEND = "sklearn"

print(f"GBM backend: {GBM_BACKEND}")

DRIFT_DATA_PATH = Path("data/drift_episodes.parquet")
DRIFT_MODEL_OUTPUT = Path("models/drift_time_xgb.pkl")
DRIFT_REPORT_OUTPUT = Path("docs/DRIFT_MODEL_PERFORMANCE.md")

MIN_DURATION = 5.0
MAX_DURATION = 480.0
SHRINKAGE_N = 10      # per-home shrinkage (same as active-time model)
RANDOM_SEED = 42

DRIFT_FEATURES = [
    "margin",
    "log_margin",
    "is_heating",
    "signed_thermal_drive",
    "outdoor_temp",
    "start_temp",
    "boundary_temp",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "hour",
    "month",
    "day_of_week",
]


# ---------------------------------------------------------------------------
# GBM factory
# ---------------------------------------------------------------------------

def _make_gbm(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10):
    if GBM_BACKEND == "lightgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
    elif GBM_BACKEND == "xgboost":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        return HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            min_samples_leaf=min_samples,
            random_state=RANDOM_SEED,
        )


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_and_featurize() -> pd.DataFrame:
    """Load drift episodes, filter, and engineer all model features."""
    print("Loading drift episodes...")
    t0 = time.time()

    df = (
        pl.scan_parquet(DRIFT_DATA_PATH)
        .filter(
            (pl.col("timestep_idx") == 0)
            & (pl.col("crossed_boundary") == True)
            & pl.col("time_to_boundary_min").is_between(MIN_DURATION, MAX_DURATION)
        )
        .select(
            "home_id",
            "split",
            "episode_id",
            "timestamp",
            "drift_direction",
            "start_temp",
            "boundary_temp",
            "distance_to_boundary",
            "Outdoor_Temperature",
            "time_to_boundary_min",
            "recent_indoor_slope_15m",
            "time_since_hvac_off_min",
        )
        .collect()
        .to_pandas()
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["duration_min"] = df["time_to_boundary_min"].astype(float)
    df["log_duration"] = np.log(df["duration_min"])

    # Direction flag
    df["is_heating"] = (df["drift_direction"] == "warming_drift").astype(float)

    # Margin: how far (°F) from the comfort boundary at episode start.
    # distance_to_boundary is already the unsigned gap from the data pipeline.
    df["margin"] = df["distance_to_boundary"].astype(float)
    df = df[df["margin"] > 1e-6].copy()
    df["log_margin"] = np.log(df["margin"] + 0.1)

    # Signed thermal drive:
    #   Heating (warming_drift): HVAC off, home cooling down.
    #   Outdoor is usually colder → drives home toward boundary faster.
    #   Convention: positive = outdoor accelerates drift toward boundary.
    #   signed_thermal_drive = start_temp - outdoor  for warming_drift
    #                        = outdoor - start_temp  for cooling_drift
    outdoor = df["Outdoor_Temperature"].astype(float)
    indoor = df["start_temp"].astype(float)
    df["signed_thermal_drive"] = np.where(
        df["is_heating"] == 1.0,
        indoor - outdoor,   # warming_drift: cold outdoor → fast cooling → positive
        outdoor - indoor,   # cooling_drift: hot outdoor → fast heating → positive
    )
    df["outdoor_temp"] = outdoor
    df["start_temp"] = indoor

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    elapsed = time.time() - t0
    print(f"  {len(df):,} episodes from {df['home_id'].nunique()} homes in {elapsed:.1f}s")
    print(f"  Direction counts: {df['drift_direction'].value_counts().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# Within-home 70/30 split (for per-home correction evaluation)
# ---------------------------------------------------------------------------

def within_home_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each home 70/30 on episode_id. Returns (train_df, test_df)."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_rows, test_rows = [], []

    for home, group in df.groupby("home_id"):
        episodes = group["episode_id"].unique()
        rng.shuffle(episodes)
        n_train = max(1, int(len(episodes) * train_frac))
        train_eps = set(episodes[:n_train])

        mask = group["episode_id"].isin(train_eps)
        train_rows.append(group[mask])
        test_rows.append(group[~mask])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_global_gbm(train_df: pd.DataFrame):
    """Fit global GBM on log-duration. Returns trained model."""
    X = train_df[DRIFT_FEATURES].fillna(0).values
    y = train_df["log_duration"].values
    model = _make_gbm()
    model.fit(X, y)
    return model


def train_global_gbm_linear_target(train_df: pd.DataFrame):
    """Fit global GBM on raw duration minutes."""
    X = train_df[DRIFT_FEATURES].fillna(0).values
    y = train_df["duration_min"].values
    model = _make_gbm()
    model.fit(X, y)
    return model


def predict_global_linear_target(model, test_df: pd.DataFrame) -> np.ndarray:
    X = test_df[DRIFT_FEATURES].fillna(0).values
    minutes = model.predict(X)
    return np.clip(minutes, MIN_DURATION, MAX_DURATION)


def predict_direction_median(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Constant baseline: median duration by drift direction, with global fallback."""
    global_median = float(train_df["duration_min"].median())
    medians = train_df.groupby("drift_direction")["duration_min"].median().to_dict()
    return np.array([
        medians.get(direction, global_median)
        for direction in test_df["drift_direction"].values
    ], dtype=float)


def predict_global_linear_rate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Simple physics baseline: duration = k_direction * margin.
    This is the drift analogue of the active-time linear rate baseline.
    """
    k_global = {}
    for direction in ["warming_drift", "cooling_drift"]:
        mask = train_df["drift_direction"] == direction
        if mask.sum() > 0:
            margins = train_df.loc[mask, "margin"].values
            times = train_df.loc[mask, "duration_min"].values
            k_global[direction] = float(np.dot(times, margins) / (np.dot(margins, margins) + 1e-8))

    fallback_k = float(train_df["duration_min"].mean() / max(train_df["margin"].mean(), 1e-6))
    pred = np.array([
        k_global.get(row.drift_direction, fallback_k) * row.margin
        for row in test_df.itertuples(index=False)
    ], dtype=float)
    return np.clip(pred, MIN_DURATION, MAX_DURATION)


def predict_per_home_linear_rate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Per-home + direction linear-scale baseline.
    Falls back to a global direction-specific rate for unseen homes or sparse slots.
    """
    k_global = {}
    for direction in ["warming_drift", "cooling_drift"]:
        mask = train_df["drift_direction"] == direction
        if mask.sum() > 0:
            margins = train_df.loc[mask, "margin"].values
            times = train_df.loc[mask, "duration_min"].values
            k_global[direction] = float(np.dot(times, margins) / (np.dot(margins, margins) + 1e-8))

    home_direction_k = {}
    for (home, direction), group in train_df.groupby(["home_id", "drift_direction"]):
        if len(group) >= 5:
            margins = group["margin"].values
            times = group["duration_min"].values
            k = float(np.dot(times, margins) / (np.dot(margins, margins) + 1e-8))
            home_direction_k[(home, direction)] = np.clip(k, MIN_DURATION, MAX_DURATION)

    fallback_k = float(train_df["duration_min"].mean() / max(train_df["margin"].mean(), 1e-6))
    pred = np.array([
        home_direction_k.get(
            (row.home_id, row.drift_direction),
            k_global.get(row.drift_direction, fallback_k),
        ) * row.margin
        for row in test_df.itertuples(index=False)
    ], dtype=float)
    return np.clip(pred, MIN_DURATION, MAX_DURATION)


def predict_log_per_home_direction(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Per-home + direction log-linear baseline:
    log(duration) = intercept + slope * log_margin.
    """
    global_params = {}
    for direction in ["warming_drift", "cooling_drift"]:
        mask = train_df["drift_direction"] == direction
        if mask.sum() > 0:
            model = Ridge(alpha=0.1, solver="lsqr")
            model.fit(train_df.loc[mask, ["log_margin"]].values, train_df.loc[mask, "log_duration"].values)
            global_params[direction] = (float(model.intercept_), float(model.coef_[0]))

    fallback = (float(train_df["log_duration"].median()), 1.0)
    home_direction_params = {}
    for (home, direction), group in train_df.groupby(["home_id", "drift_direction"]):
        if len(group) >= 5:
            model = Ridge(alpha=0.1, solver="lsqr")
            model.fit(group[["log_margin"]].values, group["log_duration"].values)
            home_direction_params[(home, direction)] = (float(model.intercept_), float(model.coef_[0]))

    pred = np.zeros(len(test_df))
    for i, row in enumerate(test_df.itertuples(index=False)):
        intercept, slope = home_direction_params.get(
            (row.home_id, row.drift_direction),
            global_params.get(row.drift_direction, fallback),
        )
        pred[i] = np.exp(intercept + slope * row.log_margin)
    return np.clip(pred, MIN_DURATION, MAX_DURATION)


def fit_enhanced_ridge(train_df: pd.DataFrame, per_home_offsets: bool) -> dict:
    """Fit an enhanced log-linear Ridge model, optionally with per-home×direction residual offsets."""
    feature_cols = [
        "log_margin",
        "margin",
        "is_heating",
        "signed_thermal_drive",
        "outdoor_temp",
        "start_temp",
        "boundary_temp",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["log_duration"].values
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(X_train, y_train)

    home_direction_offsets = {}
    if per_home_offsets:
        residuals = y_train - model.predict(X_train)
        for (home, direction), group_idx in train_df.groupby(["home_id", "drift_direction"]).groups.items():
            loc = train_df.index.get_indexer(group_idx)
            if len(loc) >= 3:
                n = len(loc)
                weight = n / (n + SHRINKAGE_N)
                home_direction_offsets[(home, direction)] = float(residuals[loc].mean() * weight)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "home_direction_offsets": home_direction_offsets,
    }


def predict_enhanced_ridge(artifact: dict, test_df: pd.DataFrame) -> np.ndarray:
    X_test = test_df[artifact["feature_cols"]].fillna(0).values
    log_pred = artifact["model"].predict(X_test)
    offsets = artifact["home_direction_offsets"]
    if offsets:
        for i, row in enumerate(test_df.itertuples(index=False)):
            log_pred[i] += offsets.get((row.home_id, row.drift_direction), 0.0)
    return np.clip(np.exp(log_pred), MIN_DURATION, MAX_DURATION)


def train_gbm_home_encoded(train_df: pd.DataFrame):
    """Fit global GBM with home and home×direction target encodings."""
    global_mean = float(train_df["log_duration"].mean())
    smoothing = SHRINKAGE_N

    home_stats = train_df.groupby("home_id")["log_duration"].agg(["mean", "count"])
    home_stats["home_encoded"] = (
        (home_stats["count"] * home_stats["mean"] + smoothing * global_mean)
        / (home_stats["count"] + smoothing)
    )

    home_direction_stats = train_df.groupby(["home_id", "drift_direction"])["log_duration"].agg(["mean", "count"])
    home_direction_stats["home_direction_encoded"] = (
        (home_direction_stats["count"] * home_direction_stats["mean"] + smoothing * global_mean)
        / (home_direction_stats["count"] + smoothing)
    )

    def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
        encoded = df.copy()
        encoded["home_encoded"] = encoded["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)
        encoded["home_direction_encoded"] = [
            home_direction_stats["home_direction_encoded"].get((row.home_id, row.drift_direction), global_mean)
            for row in encoded.itertuples(index=False)
        ]
        return encoded

    feature_cols = DRIFT_FEATURES + ["home_encoded", "home_direction_encoded"]
    train_encoded = add_encodings(train_df)
    model = _make_gbm()
    model.fit(train_encoded[feature_cols].fillna(0).values, train_encoded["log_duration"].values)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "home_stats": home_stats,
        "home_direction_stats": home_direction_stats,
        "global_mean": global_mean,
    }


def predict_gbm_home_encoded(artifact: dict, test_df: pd.DataFrame) -> np.ndarray:
    encoded = test_df.copy()
    home_stats = artifact["home_stats"]
    home_direction_stats = artifact["home_direction_stats"]
    global_mean = artifact["global_mean"]
    encoded["home_encoded"] = encoded["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)
    encoded["home_direction_encoded"] = [
        home_direction_stats["home_direction_encoded"].get((row.home_id, row.drift_direction), global_mean)
        for row in encoded.itertuples(index=False)
    ]
    log_pred = artifact["model"].predict(encoded[artifact["feature_cols"]].fillna(0).values)
    return np.clip(np.exp(log_pred), MIN_DURATION, MAX_DURATION)


def train_per_home_gbms(train_df: pd.DataFrame) -> dict:
    """Fit smaller per-home GBMs with a global fallback."""
    X_train = train_df[DRIFT_FEATURES].fillna(0).values
    y_train = train_df["log_duration"].values
    global_model = _make_gbm(max_depth=4, n_estimators=100, min_samples=5)
    global_model.fit(X_train, y_train)

    home_models = {}
    for home, group in train_df.groupby("home_id"):
        if len(group) >= 10:
            model = _make_gbm(max_depth=3, n_estimators=50, min_samples=3)
            model.fit(group[DRIFT_FEATURES].fillna(0).values, group["log_duration"].values)
            home_models[home] = model

    return {
        "global_model": global_model,
        "home_models": home_models,
    }


def predict_per_home_gbms(artifact: dict, test_df: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(test_df))
    for home, group_idx in test_df.groupby("home_id").groups.items():
        model = artifact["home_models"].get(home, artifact["global_model"])
        positions = test_df.index.get_indexer(group_idx)
        X = test_df.loc[group_idx, DRIFT_FEATURES].fillna(0).values
        pred[positions] = np.exp(model.predict(X))
    return np.clip(pred, MIN_DURATION, MAX_DURATION)


def compute_home_residuals(
    model,
    train_df: pd.DataFrame,
    shrinkage_n: int = SHRINKAGE_N,
) -> tuple[dict, dict]:
    """
    Compute per-home (and per-home×mode) mean residuals on training data.
    Returns (home_residuals, home_mode_residuals), both shrunk toward 0.
    """
    X_train = train_df[DRIFT_FEATURES].fillna(0).values
    preds = model.predict(X_train)
    residuals = train_df["log_duration"].values - preds

    home_residuals: dict[str, float] = {}
    for home, group_idx in train_df.groupby("home_id").groups.items():
        r = residuals[train_df.index.get_indexer(group_idx)]
        n = len(r)
        weight = n / (n + shrinkage_n)
        home_residuals[home] = float(r.mean() * weight)

    home_mode_residuals: dict[tuple[str, int], float] = {}
    for (home, mode), group_idx in train_df.groupby(["home_id", "is_heating"]).groups.items():
        r = residuals[train_df.index.get_indexer(group_idx)]
        n = len(r)
        if n >= 3:
            weight = n / (n + shrinkage_n)
            home_mode_residuals[(home, int(mode))] = float(r.mean() * weight)

    return home_residuals, home_mode_residuals


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def predict_global(model, test_df: pd.DataFrame) -> np.ndarray:
    X = test_df[DRIFT_FEATURES].fillna(0).values
    log_pred = model.predict(X)
    minutes = np.exp(log_pred)
    return np.clip(minutes, MIN_DURATION, MAX_DURATION)


def predict_with_home_correction(
    model,
    test_df: pd.DataFrame,
    home_residuals: dict,
    home_mode_residuals: dict,
) -> np.ndarray:
    X = test_df[DRIFT_FEATURES].fillna(0).values
    log_preds = model.predict(X)

    for i, (_, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        mode = int(row["is_heating"])
        if (home, mode) in home_mode_residuals:
            log_preds[i] += home_mode_residuals[(home, mode)]
        elif home in home_residuals:
            log_preds[i] += home_residuals[home]

    minutes = np.exp(log_preds)
    return np.clip(minutes, MIN_DURATION, MAX_DURATION)


def compute_metrics(pred: np.ndarray, actual: np.ndarray, label: str = "") -> dict:
    errors = pred - actual
    abs_errors = np.abs(errors)
    metrics = {
        "label": label,
        "n": len(pred),
        "mae": float(abs_errors.mean()),
        "median_ae": float(np.median(abs_errors)),
        "p90": float(np.percentile(abs_errors, 90)),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "bias": float(errors.mean()),
    }
    return metrics


def compute_stratified(pred: np.ndarray, actual: np.ndarray, df: pd.DataFrame) -> dict:
    """Compute metrics broken out by direction and margin bucket."""
    strats = {}

    # By direction
    for direction in ["warming_drift", "cooling_drift"]:
        mask = df["drift_direction"].values == direction
        if mask.sum() > 0:
            strats[direction] = compute_metrics(pred[mask], actual[mask], direction)

    # By margin bucket
    margin = df["margin"].values
    for label, lo, hi in [("1–2°F", 1, 2), ("2–3°F", 2, 3), ("3–5°F", 3, 5), (">5°F", 5, 999)]:
        mask = (margin >= lo) & (margin < hi)
        if mask.sum() > 0:
            strats[label] = compute_metrics(pred[mask], actual[mask], label)

    # By duration bucket (easy/hard)
    dur = actual
    for label, lo, hi in [("≤30 min", 0, 30), ("30–120 min", 30, 120), (">120 min", 120, 9999)]:
        mask = (dur > lo) & (dur <= hi)
        if mask.sum() > 0:
            strats[label] = compute_metrics(pred[mask], actual[mask], label)

    return strats


# ---------------------------------------------------------------------------
# Artifact building and saving
# ---------------------------------------------------------------------------

def build_artifact(
    model,
    home_residuals: dict,
    home_mode_residuals: dict,
    n_train_rows: int,
) -> dict:
    return {
        "version": 2,
        "model_type": "gbm_drift_time_to_boundary",
        "backend": GBM_BACKEND,
        "model": model,
        "feature_cols": DRIFT_FEATURES,
        "predict_log": True,
        "min_duration": MIN_DURATION,
        "max_duration": MAX_DURATION,
        "n_train_rows": n_train_rows,
        "home_residuals": home_residuals,
        "home_mode_residuals": home_mode_residuals,
        "residual_shrinkage_n": SHRINKAGE_N,
        "n_homes_with_correction": len(home_residuals),
        "n_home_modes_with_correction": len(home_mode_residuals),
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def fmt_metrics_row(m: dict) -> str:
    return f"| {m['label']:<38} | {m['n']:>8,} | {m['mae']:>6.1f} | {m['median_ae']:>6.1f} | {m['p90']:>6.1f} | {m['rmse']:>6.1f} | {m['bias']:>+6.1f} |"


def write_report(
    cross_results: list[dict],
    within_results: list[dict],
    cross_strats: dict,
    within_strats: dict,
    artifact: dict,
    output_path: Path,
) -> None:
    cross_by_label = {m["label"]: m for m in cross_results}
    within_by_label = {m["label"]: m for m in within_results}
    cross_linear = cross_by_label.get("Global Linear Rate", {})
    cross_enhanced = cross_by_label.get("Enhanced Ridge (global)", {})
    cross_global = cross_by_label.get("Global GBM (cross-home, test homes)", {})
    within_global = within_by_label.get("Global GBM (within-home)", {})
    within_home_encoded = within_by_label.get("Global GBM + Home Encoding", {})
    within_per_home_gbm = within_by_label.get("Per-Home GBM", {})
    within_hybrid = within_by_label.get("Global + Per-Home Correction", {})

    lines = []

    lines += [
        "# Drift Model Performance",
        "",
        "Evaluation of the passive thermal drift time-to-boundary model (`models/drift_time_xgb.pkl`).",
        "Predicts how many minutes before indoor temperature passively drifts from the current value",
        "to a comfort boundary, with HVAC off.",
        "",
        "---",
        "",
        "## Model Summary",
        "",
        f"- **Architecture**: Gradient Boosted Trees ({artifact['backend']})",
        f"- **Training rows** (cross-home): {artifact['n_train_rows']:,}",
        f"- **Features**: {len(DRIFT_FEATURES)} (margin, log_margin, thermal drive, time/season encoding, temps)",
        f"- **Target**: log(minutes to boundary) → exponentiated and clipped to [{MIN_DURATION:.0f}, {MAX_DURATION:.0f}] min",
        f"- **Per-home homes covered**: {artifact['n_homes_with_correction']:,}",
        f"- **Per-home×mode slots covered**: {artifact['n_home_modes_with_correction']:,}",
        f"- **Residual shrinkage N**: {artifact['residual_shrinkage_n']}",
        "",
        "---",
        "",
        "## Evaluation Setup",
        "",
        "Two complementary evaluations are reported:",
        "",
        "### 1. Cross-Home (train → test homes)",
        "",
        "Uses the project's canonical home-level split (`split` column in `drift_episodes.parquet`).",
        "Train homes: 679 · Val homes: 96 · Test homes: 195.",
        "Test homes are **completely unseen** during training — this measures true generalization.",
        "Per-home correction cannot apply to test homes (no residual history).",
        "",
        "### 2. Within-Home (70/30 per home)",
        "",
        "Each home's episodes are split 70/30 randomly. The same home appears in both train and test.",
        "This allows the per-home residual correction to be applied on the 30% test portion.",
        "Shows the best-case benefit of per-home correction for a **known** home.",
        "",
        "---",
        "",
        "## Results: Cross-Home Evaluation",
        "",
        "| Model                                   |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for m in cross_results:
        lines.append(fmt_metrics_row(m))

    lines += [
        "",
        "Per-home and home-encoded methods are omitted here when they require prior home history.",
        "Test homes are unseen, so these results are the cold-start benchmark set.",
        "",
        "---",
        "",
        "## Results: Within-Home Evaluation",
        "",
        "| Model                                   |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for m in within_results:
        lines.append(fmt_metrics_row(m))

    lines += [
        "",
        "---",
        "",
        "## Stratified Breakdown (Cross-Home, Global Model)",
        "",
        "### By Direction",
        "",
        "| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for key in ["warming_drift", "cooling_drift"]:
        if key in cross_strats:
            lines.append(fmt_metrics_row(cross_strats[key]))

    lines += [
        "",
        "### By Margin (distance from boundary at episode start)",
        "",
        "| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for key in ["1–2°F", "2–3°F", "3–5°F", ">5°F"]:
        if key in cross_strats:
            lines.append(fmt_metrics_row(cross_strats[key]))

    lines += [
        "",
        "### By True Duration",
        "",
        "| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for key in ["≤30 min", "30–120 min", ">120 min"]:
        if key in cross_strats:
            lines.append(fmt_metrics_row(cross_strats[key]))

    lines += [
        "",
        "---",
        "",
        "## Stratified Breakdown (Within-Home, Global + Per-Home)",
        "",
        "### By Direction",
        "",
        "| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for key in ["warming_drift", "cooling_drift"]:
        if key in within_strats:
            lines.append(fmt_metrics_row(within_strats[key]))

    lines += [
        "",
        "### By Margin",
        "",
        "| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for key in ["1–2°F", "2–3°F", "3–5°F", ">5°F"]:
        if key in within_strats:
            lines.append(fmt_metrics_row(within_strats[key]))

    lines += [
        "",
        "### By True Duration",
        "",
        "| Stratum                                 |        N |    MAE | Median |    P90 |   RMSE |   Bias |",
        "|-----------------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for key in ["≤30 min", "30–120 min", ">120 min"]:
        if key in within_strats:
            lines.append(fmt_metrics_row(within_strats[key]))

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Benchmark Context",
        "",
        "The drift evaluator now mirrors the active-time benchmark ladder: simple direction medians,",
        "linear rate models, log-duration linear models, enhanced Ridge models, global GBM variants,",
        "per-home GBMs, and the hybrid global + per-home residual model. Cross-home evaluation only",
        "uses methods that can operate for unseen homes without prior history; within-home evaluation",
        "adds the per-home and home-encoded variants.",
        "",
        f"In the cold-start cross-home setup, the global linear rate baseline is {cross_linear.get('mae', float('nan')):.1f} min MAE,",
        f"the enhanced Ridge model is {cross_enhanced.get('mae', float('nan')):.1f} min MAE, and the global GBM remains best at",
        f"{cross_global.get('mae', float('nan')):.1f} min MAE.",
        "",
        f"In the known-home setup, home information matters: home encoding improves the global GBM from",
        f"{within_global.get('mae', float('nan')):.1f} to {within_home_encoded.get('mae', float('nan')):.1f} min MAE,",
        f"per-home GBM reaches {within_per_home_gbm.get('mae', float('nan')):.1f} min MAE, and the deployed-style hybrid",
        f"residual correction reaches {within_hybrid.get('mae', float('nan')):.1f} min MAE.",
        "",
        "### Cross-Home vs. Within-Home Gap",
        "",
        "The cross-home MAE measures how well the model generalizes to a brand-new home with no",
        "prior history. The within-home MAE (global) measures performance when the model has seen",
        "other episodes from the same home. The difference between these two numbers is the",
        "\"cold-start penalty\" — the cost of not having any home-specific data.",
        "",
        "### Per-Home Correction Benefit",
        "",
        "Comparing within-home global vs. within-home per-home corrected shows how much per-home",
        "residuals improve predictions for **known** homes. The correction is applied in log-space",
        "with shrinkage (N=10): homes with fewer episodes get corrections pulled toward 0.",
        "",
        "### Margin Is the Primary Driver",
        "",
        "Drift time scales almost linearly with margin (°F from boundary). The `log_margin` feature",
        "captures this. Predictions for small margins (1–2°F) are hardest: a 1°F error in start_temp",
        "due to Ecobee's integer quantization can shift the margin by 100% and the predicted time",
        "substantially.",
        "",
        "### Direction Asymmetry",
        "",
        "Warming drift (heating season) tends to have longer times to boundary — homes in winter",
        "are pre-heated to near the upper comfort bound before peak, so the boundary (lower comfort",
        "bound) is often 4°F away. Cooling drift has smaller margins on average. Expect MAE to",
        "differ between the two.",
        "",
        "---",
        "",
        "## Artifact Details",
        "",
        "**Path**: `models/drift_time_xgb.pkl`",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| `version` | {artifact['version']} |",
        f"| `model_type` | `{artifact['model_type']}` |",
        f"| `backend` | `{artifact['backend']}` |",
        f"| `n_train_rows` | {artifact['n_train_rows']:,} |",
        f"| `n_homes_with_correction` | {artifact['n_homes_with_correction']:,} |",
        f"| `n_home_modes_with_correction` | {artifact['n_home_modes_with_correction']:,} |",
        f"| `min_duration` | {artifact['min_duration']} min |",
        f"| `max_duration` | {artifact['max_duration']} min |",
        "",
        "The artifact now includes `home_residuals` and `home_mode_residuals` dicts,",
        "matching the structure of the active-time artifact. `predict_drift_time()` in",
        "`mpc/model_interfaces.py` applies per-home correction when `home_id` is provided",
        "and the home is in the training set.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(save: bool = True) -> None:
    print("=" * 65)
    print("Drift Model Evaluation")
    print("=" * 65)

    # Load data
    df = load_and_featurize()

    # ------------------------------------------------------------------
    # Part 1: Cross-home evaluation (canonical home-level split)
    # ------------------------------------------------------------------
    print("\n[1/2] Cross-home evaluation (train split → test split)")
    print("-" * 65)

    train_cross = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_cross = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"  Train: {len(train_cross):,} episodes from {train_cross['home_id'].nunique()} homes")
    print(f"  Test:  {len(test_cross):,} episodes from {test_cross['home_id'].nunique()} homes")

    print("  Running cross-home benchmarks...")
    t0 = time.time()
    pred_cross_median = predict_direction_median(train_cross, test_cross)
    pred_cross_linear = predict_global_linear_rate(train_cross, test_cross)

    enhanced_cross_artifact = fit_enhanced_ridge(train_cross, per_home_offsets=False)
    pred_cross_enhanced = predict_enhanced_ridge(enhanced_cross_artifact, test_cross)

    print("    Training global GBM (log target)...")
    global_model_cross = train_global_gbm(train_cross)
    pred_cross = predict_global(global_model_cross, test_cross)

    print("    Training global GBM (linear target)...")
    global_linear_model_cross = train_global_gbm_linear_target(train_cross)
    pred_cross_gbm_linear = predict_global_linear_target(global_linear_model_cross, test_cross)
    print(f"  Done in {time.time() - t0:.1f}s")

    actual_cross = test_cross["duration_min"].values

    cross_median = compute_metrics(pred_cross_median, actual_cross, "Direction Median")
    cross_linear = compute_metrics(pred_cross_linear, actual_cross, "Global Linear Rate")
    cross_enhanced = compute_metrics(pred_cross_enhanced, actual_cross, "Enhanced Ridge (global)")
    cross_global = compute_metrics(pred_cross, actual_cross, "Global GBM (cross-home, test homes)")
    cross_gbm_linear = compute_metrics(pred_cross_gbm_linear, actual_cross, "Global GBM (linear target)")

    cross_results = [
        cross_median,
        cross_linear,
        cross_enhanced,
        cross_global,
        cross_gbm_linear,
    ]
    cross_strats = compute_stratified(pred_cross, actual_cross, test_cross)

    print(f"\n  Cross-home results:")
    for m in cross_results:
        print(f"    {m['label']:<36} MAE={m['mae']:.1f} min  Median={m['median_ae']:.1f}  P90={m['p90']:.1f}  Bias={m['bias']:+.1f}")

    # Compute residuals from the cross-home training set for the final artifact
    print("\n  Computing per-home residuals from train set...")
    home_res_cross, home_mode_res_cross = compute_home_residuals(global_model_cross, train_cross)
    print(f"    {len(home_res_cross)} home residuals  |  {len(home_mode_res_cross)} home×mode residuals")
    residual_stds = np.std(list(home_res_cross.values()))
    print(f"    Residual std (log-space): {residual_stds:.3f}  (~{100*(np.exp(residual_stds)-1):.0f}% multiplicative spread)")

    # ------------------------------------------------------------------
    # Part 2: Within-home evaluation (per-home correction benefit)
    # ------------------------------------------------------------------
    print("\n[2/2] Within-home evaluation (70/30 per home, train homes only)")
    print("-" * 65)

    print("  Splitting train homes 70/30 within each home...")
    train_within, test_within = within_home_split(train_cross, train_frac=0.70)
    print(f"  Train: {len(train_within):,} episodes  |  Test: {len(test_within):,} episodes")

    print("  Running within-home benchmarks...")
    t0 = time.time()
    pred_within_median = predict_direction_median(train_within, test_within)
    pred_within_linear = predict_global_linear_rate(train_within, test_within)
    pred_within_per_home_linear = predict_per_home_linear_rate(train_within, test_within)
    pred_within_log_per_home = predict_log_per_home_direction(train_within, test_within)

    enhanced_within_artifact = fit_enhanced_ridge(train_within, per_home_offsets=True)
    pred_within_enhanced = predict_enhanced_ridge(enhanced_within_artifact, test_within)

    print("    Training global GBM (log target)...")
    global_model_within = train_global_gbm(train_within)
    pred_within_global = predict_global(global_model_within, test_within)

    print("    Training global GBM (linear target)...")
    global_linear_model_within = train_global_gbm_linear_target(train_within)
    pred_within_gbm_linear = predict_global_linear_target(global_linear_model_within, test_within)

    print("    Training global GBM + home target encoding...")
    home_encoded_artifact = train_gbm_home_encoded(train_within)
    pred_within_home_encoded = predict_gbm_home_encoded(home_encoded_artifact, test_within)

    print("    Training per-home GBMs...")
    per_home_gbm_artifact = train_per_home_gbms(train_within)
    pred_within_per_home_gbm = predict_per_home_gbms(per_home_gbm_artifact, test_within)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("  Computing per-home residuals...")
    home_res_within, home_mode_res_within = compute_home_residuals(global_model_within, train_within)
    print(f"    {len(home_res_within)} home residuals  |  {len(home_mode_res_within)} home×mode residuals")

    pred_within_hybrid = predict_with_home_correction(
        global_model_within, test_within, home_res_within, home_mode_res_within
    )
    actual_within = test_within["duration_min"].values

    within_median = compute_metrics(pred_within_median, actual_within, "Direction Median")
    within_linear = compute_metrics(pred_within_linear, actual_within, "Global Linear Rate")
    within_per_home_linear = compute_metrics(pred_within_per_home_linear, actual_within, "Per-Home Linear Rate")
    within_log_per_home = compute_metrics(pred_within_log_per_home, actual_within, "Log-Duration Per-Home")
    within_enhanced = compute_metrics(pred_within_enhanced, actual_within, "Enhanced Ridge + Per-Home")
    within_global = compute_metrics(pred_within_global, actual_within, "Global GBM (within-home)")
    within_gbm_linear = compute_metrics(pred_within_gbm_linear, actual_within, "Global GBM (linear target)")
    within_home_encoded = compute_metrics(pred_within_home_encoded, actual_within, "Global GBM + Home Encoding")
    within_per_home_gbm = compute_metrics(pred_within_per_home_gbm, actual_within, "Per-Home GBM")
    within_hybrid = compute_metrics(pred_within_hybrid, actual_within, "Global + Per-Home Correction")

    within_results = [
        within_median,
        within_linear,
        within_per_home_linear,
        within_log_per_home,
        within_enhanced,
        within_global,
        within_gbm_linear,
        within_home_encoded,
        within_per_home_gbm,
        within_hybrid,
    ]
    within_strats = compute_stratified(pred_within_hybrid, actual_within, test_within)

    improvement = 100 * (within_global["mae"] - within_hybrid["mae"]) / within_global["mae"]
    print(f"\n  Within-home results:")
    for m in within_results:
        print(f"    {m['label']:<36} MAE={m['mae']:.1f} min  Median={m['median_ae']:.1f}  P90={m['p90']:.1f}  Bias={m['bias']:+.1f}")
    print(f"    Hybrid vs. global GBM: {improvement:+.1f}% MAE")

    # ------------------------------------------------------------------
    # Build and save the final artifact (global model from cross-home training)
    # ------------------------------------------------------------------
    artifact = build_artifact(
        model=global_model_cross,
        home_residuals=home_res_cross,
        home_mode_residuals=home_mode_res_cross,
        n_train_rows=len(train_cross),
    )

    if save:
        DRIFT_MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with DRIFT_MODEL_OUTPUT.open("wb") as f:
            pickle.dump(artifact, f)
        print(f"\nSaved artifact: {DRIFT_MODEL_OUTPUT}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Model':<42} {'MAE':>6} {'Median':>6} {'P90':>6} {'Bias':>6}")
    print("-" * 65)
    for m in cross_results + within_results:
        print(f"{m['label']:<42} {m['mae']:>6.1f} {m['median_ae']:>6.1f} {m['p90']:>6.1f} {m['bias']:>+6.1f}")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    write_report(
        cross_results=cross_results,
        within_results=within_results,
        cross_strats=cross_strats,
        within_strats=within_strats,
        artifact=artifact,
        output_path=DRIFT_REPORT_OUTPUT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drift time-to-boundary model")
    parser.add_argument("--no-save", action="store_true", help="Don't overwrite model artifact")
    args = parser.parse_args()
    main(save=not args.no_save)
