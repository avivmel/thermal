"""
Gradient Boosting baselines for time-to-target prediction.

Implements:
1. Global GBM (all homes, no home info)
2. Global GBM with home target encoding
3. Per-home GBM (separate model per home with global fallback)
4. Hybrid GBM (global + per-home residual correction)

Uses LightGBM if available, falls back to sklearn HistGradientBoosting.

Usage: python scripts/run_xgboost_baselines.py
"""

import argparse
import pickle
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")

# Try LightGBM first, then XGBoost, then sklearn
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

print(f"Using GBM backend: {GBM_BACKEND}")

# Config
DATA_PATH = "data/setpoint_responses.parquet"
DRIFT_DATA_PATH = "data/drift_episodes.parquet"
DEFAULT_MODEL_OUTPUT = Path("models/active_time_xgb.pkl")
DEFAULT_DRIFT_MODEL_OUTPUT = Path("models/drift_time_xgb.pkl")
MIN_DURATION = 5    # Minimum duration in minutes (1 timestep)
MAX_DURATION = 240  # Maximum duration in minutes (4 hours) - filter anomalies
MIN_DRIFT_DURATION = 5
MAX_DRIFT_DURATION = 480
MIN_EPISODES_PER_HOME = 10  # Minimum episodes to train per-home model


@dataclass
class StratMetrics:
    """Metrics for a stratum."""
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    bias: float = 0.0
    n_samples: int = 0


@dataclass
class Metrics:
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    median_ae: float = 0.0
    p90: float = 0.0
    bias: float = 0.0
    by_type: Dict[str, StratMetrics] = field(default_factory=dict)
    by_gap: Dict[str, StratMetrics] = field(default_factory=dict)
    by_system: Dict[str, StratMetrics] = field(default_factory=dict)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    change_types: np.ndarray,
    initial_gaps: np.ndarray,
    system_running: np.ndarray = None,
) -> Metrics:
    """Compute all evaluation metrics."""
    errors = pred - target
    abs_errors = np.abs(errors)

    valid_target = target > 1

    m = Metrics()
    m.mae = abs_errors.mean()
    m.rmse = np.sqrt((errors ** 2).mean())
    m.mape = 100 * (abs_errors[valid_target] / target[valid_target]).mean() if valid_target.sum() > 0 else 0
    m.median_ae = np.median(abs_errors)
    m.p90 = np.percentile(abs_errors, 90)
    m.bias = errors.mean()

    # By change type
    for ctype in ["heat_increase", "cool_decrease"]:
        mask = change_types == ctype
        if mask.sum() > 0:
            type_errors = pred[mask] - target[mask]
            valid = target[mask] > 1
            m.by_type[ctype] = StratMetrics(
                mae=np.abs(type_errors).mean(),
                rmse=np.sqrt((type_errors ** 2).mean()),
                mape=100 * (np.abs(type_errors[valid]) / target[mask][valid]).mean() if valid.sum() > 0 else 0,
                bias=type_errors.mean(),
                n_samples=mask.sum(),
            )

    # By initial gap size
    gap_bins = [
        ("1-2°F", 1, 2),
        ("2-3°F", 2, 3),
        ("3-5°F", 3, 5),
        (">5°F", 5, 100),
    ]
    for name, lo, hi in gap_bins:
        mask = (np.abs(initial_gaps) >= lo) & (np.abs(initial_gaps) < hi)
        if mask.sum() > 0:
            gap_errors = pred[mask] - target[mask]
            valid = target[mask] > 1
            m.by_gap[name] = StratMetrics(
                mae=np.abs(gap_errors).mean(),
                rmse=np.sqrt((gap_errors ** 2).mean()),
                mape=100 * (np.abs(gap_errors[valid]) / target[mask][valid]).mean() if valid.sum() > 0 else 0,
                bias=gap_errors.mean(),
                n_samples=mask.sum(),
            )

    # By system running
    if system_running is not None:
        for state, label in [(1, "running"), (0, "not_running")]:
            mask = system_running == state
            if mask.sum() > 0:
                state_errors = pred[mask] - target[mask]
                m.by_system[label] = StratMetrics(
                    mae=np.abs(state_errors).mean(),
                    rmse=np.sqrt((state_errors ** 2).mean()),
                    bias=state_errors.mean(),
                    n_samples=mask.sum(),
                )

    return m


def load_data() -> pl.DataFrame:
    """Load setpoint response data."""
    print("Loading data...")
    t0 = time.time()
    df = pl.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def create_episode_samples(df: pl.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create episode-level samples for time-to-target prediction.
    Same logic as run_improved_baselines.py for fair comparison.
    """
    print(f"Creating episode-level samples (within-home split, {train_frac:.0%} train, max {MAX_DURATION} min)...")
    t0 = time.time()

    np.random.seed(42)

    records = []
    homes = df["home_id"].unique().to_list()

    for home in tqdm(homes, desc="  Processing homes"):
        home_df = df.filter(pl.col("home_id") == home)
        episodes = home_df["episode_id"].unique().to_list()

        if len(episodes) < 2:
            continue

        np.random.shuffle(episodes)
        n_train = max(1, int(len(episodes) * train_frac))
        train_episodes = set(episodes[:n_train])

        for ep_id in episodes:
            ep = home_df.filter(pl.col("episode_id") == ep_id).sort("timestep_idx")

            if len(ep) < 1:
                continue

            n_steps = len(ep)
            duration_min = n_steps * 5

            first_row = ep[0]

            start_temp = first_row["Indoor_AverageTemperature"].item()
            target_sp = first_row["target_setpoint"].item()
            outdoor_temp = first_row["Outdoor_Temperature"].item()
            initial_gap = first_row["initial_gap"].item()
            change_type = first_row["change_type"].item()
            state = first_row["state"].item()
            timestamp = first_row["timestamp"].item()

            heat_runtime = first_row["HeatingEquipmentStage1_RunTime"].item()
            cool_runtime = first_row["CoolingEquipmentStage1_RunTime"].item()
            fan_runtime = first_row["Fan_RunTime"].item()

            indoor_humidity = first_row["Indoor_Humidity"].item()
            outdoor_humidity = first_row["Outdoor_Humidity"].item()

            if pd.isna(start_temp) or pd.isna(outdoor_temp) or pd.isna(initial_gap):
                continue
            if duration_min < MIN_DURATION or duration_min > MAX_DURATION:
                continue

            thermal_drive = outdoor_temp - start_temp

            if change_type == "heat_increase":
                signed_thermal_drive = thermal_drive
            else:
                signed_thermal_drive = -thermal_drive

            hour = timestamp.hour
            month = timestamp.month
            day_of_week = timestamp.weekday()

            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            heat_runtime = 0 if pd.isna(heat_runtime) else heat_runtime
            cool_runtime = 0 if pd.isna(cool_runtime) else cool_runtime
            fan_runtime = 0 if pd.isna(fan_runtime) else fan_runtime

            system_running = 1 if (heat_runtime > 0 or cool_runtime > 0) else 0

            records.append({
                "home_id": home,
                "episode_id": ep_id,
                "is_train": ep_id in train_episodes,
                "state": state,
                "change_type": change_type,
                "duration_min": duration_min,
                "log_duration": np.log(duration_min),
                # Core features
                "initial_gap": initial_gap,
                "abs_gap": np.abs(initial_gap),
                "log_gap": np.log(np.abs(initial_gap) + 0.1),
                "start_temp": start_temp,
                "target_setpoint": target_sp,
                "outdoor_temp": outdoor_temp,
                "thermal_drive": thermal_drive,
                "signed_thermal_drive": signed_thermal_drive,
                # Time features
                "hour": hour,
                "month": month,
                "day_of_week": day_of_week,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                # Humidity
                "indoor_humidity": indoor_humidity if not pd.isna(indoor_humidity) else 50,
                "outdoor_humidity": outdoor_humidity if not pd.isna(outdoor_humidity) else 50,
                # Runtime features
                "heat_runtime_start": heat_runtime,
                "cool_runtime_start": cool_runtime,
                "system_running": system_running,
                # Mode indicator
                "is_heating": 1 if change_type == "heat_increase" else 0,
            })

    all_df = pd.DataFrame(records)
    train_df = all_df[all_df["is_train"]].copy()
    test_df = all_df[~all_df["is_train"]].copy()

    print(f"  Train: {len(train_df):,} episodes")
    print(f"  Test: {len(test_df):,} episodes")
    print(f"  Homes in both: {len(set(train_df['home_id']) & set(test_df['home_id']))}")
    print(f"  Time: {time.time()-t0:.1f}s")

    print(f"\n  Duration stats:")
    print(f"    Train - Mean: {train_df['duration_min'].mean():.1f} min, Median: {train_df['duration_min'].median():.1f} min")
    print(f"    Test  - Mean: {test_df['duration_min'].mean():.1f} min, Median: {test_df['duration_min'].median():.1f} min")

    return train_df, test_df


# =============================================================================
# Feature columns
# =============================================================================

CORE_FEATURES = [
    "log_gap",
    "abs_gap",
    "is_heating",
    "system_running",
    "signed_thermal_drive",
    "outdoor_temp",
    "start_temp",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "hour", "month", "day_of_week",
]

HUMIDITY_FEATURES = [
    "indoor_humidity",
    "outdoor_humidity",
]

ALL_FEATURES = CORE_FEATURES + TIME_FEATURES + HUMIDITY_FEATURES

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


# =============================================================================
# GBM Model Factory
# =============================================================================

def create_gbm_model(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10):
    """Create a GBM model using the available backend."""
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
            random_state=42,
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
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:  # sklearn
        return HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            min_samples_leaf=min_samples,
            random_state=42,
        )


def get_feature_importance(model, feature_cols):
    """Get feature importance from model."""
    if GBM_BACKEND == "sklearn":
        # sklearn doesn't have feature_importances_ for all models
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_cols, model.feature_importances_))
        return {}
    return dict(zip(feature_cols, model.feature_importances_))


# =============================================================================
# Model 1: Global GBM (no home info)
# =============================================================================

def model_global_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     predict_log: bool = True) -> tuple[np.ndarray, dict]:
    """
    Global GBM model trained on all homes.
    No home-specific information - tests pure feature learning.
    """
    print(f"  Fitting global GBM (predict_log={predict_log}, backend={GBM_BACKEND})...")

    feature_cols = ALL_FEATURES
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values

    if predict_log:
        y_train = train_df["log_duration"].values
    else:
        y_train = train_df["duration_min"].values

    model = create_gbm_model(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    if predict_log:
        pred = np.exp(pred)

    # Feature importance
    importance = get_feature_importance(model, feature_cols)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {
        "n_features": len(feature_cols),
        "predict_log": predict_log,
        "top_features": top_features,
        "backend": GBM_BACKEND,
    }

    if top_features:
        print(f"    Top features: {[f'{k}:{v:.3f}' for k, v in top_features]}")

    return pred, info


# =============================================================================
# Model 2: Global XGBoost with home target encoding
# =============================================================================

def model_global_xgb_home_encoded(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Global XGBoost with home-level target encoding.

    Encodes home_id as the mean log-duration for that home (with smoothing).
    This gives the model per-home information without categorical features.
    """
    print("  Fitting global XGBoost with home target encoding...")

    # Compute home-level target encoding (mean log-duration per home)
    global_mean = train_df["log_duration"].mean()
    smoothing = 10  # Shrinkage toward global mean

    home_stats = train_df.groupby("home_id").agg({
        "log_duration": ["mean", "count"]
    })
    home_stats.columns = ["home_mean", "home_count"]
    home_stats["home_encoded"] = (
        (home_stats["home_count"] * home_stats["home_mean"] + smoothing * global_mean) /
        (home_stats["home_count"] + smoothing)
    )

    # Also encode home × mode interaction
    home_mode_stats = train_df.groupby(["home_id", "is_heating"]).agg({
        "log_duration": ["mean", "count"]
    })
    home_mode_stats.columns = ["hm_mean", "hm_count"]
    home_mode_stats["home_mode_encoded"] = (
        (home_mode_stats["hm_count"] * home_mode_stats["hm_mean"] + smoothing * global_mean) /
        (home_mode_stats["hm_count"] + smoothing)
    )

    # Add encodings to data
    train_data = train_df.copy()
    test_data = test_df.copy()

    train_data["home_encoded"] = train_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)
    test_data["home_encoded"] = test_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)

    # Home × mode encoding
    train_data["home_mode_key"] = train_data["home_id"] + "_" + train_data["is_heating"].astype(str)
    test_data["home_mode_key"] = test_data["home_id"] + "_" + test_data["is_heating"].astype(str)

    hm_encoding = home_mode_stats["home_mode_encoded"].to_dict()

    def get_hm_encoding(row):
        key = (row["home_id"], row["is_heating"])
        return hm_encoding.get(key, global_mean)

    train_data["home_mode_encoded"] = train_data.apply(get_hm_encoding, axis=1)
    test_data["home_mode_encoded"] = test_data.apply(get_hm_encoding, axis=1)

    # Feature columns
    feature_cols = ALL_FEATURES + ["home_encoded", "home_mode_encoded"]

    X_train = train_data[feature_cols].fillna(0).values
    X_test = test_data[feature_cols].fillna(0).values
    y_train = train_data["log_duration"].values

    model = create_gbm_model(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10)
    model.fit(X_train, y_train)

    pred = np.exp(model.predict(X_test))

    importance = get_feature_importance(model, feature_cols)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {
        "n_features": len(feature_cols),
        "n_homes_encoded": len(home_stats),
        "top_features": top_features,
        "home_encoded_importance": importance.get("home_encoded", 0),
        "home_mode_encoded_importance": importance.get("home_mode_encoded", 0),
    }

    if importance:
        print(f"    Home encoding importance: {info['home_encoded_importance']:.3f}")
        print(f"    Home×mode encoding importance: {info['home_mode_encoded_importance']:.3f}")
        print(f"    Top features: {[f'{k}:{v:.3f}' for k, v in top_features]}")

    return pred, info


# =============================================================================
# Model 3: Per-Home XGBoost
# =============================================================================

def model_per_home_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Separate XGBoost model per home.

    Falls back to global model for homes with insufficient training data.
    """
    print(f"  Fitting per-home XGBoost (min {MIN_EPISODES_PER_HOME} episodes)...")

    feature_cols = ALL_FEATURES

    # First train global fallback model
    X_train_all = train_df[feature_cols].fillna(0).values
    y_train_all = train_df["log_duration"].values

    global_model = create_gbm_model(max_depth=4, n_estimators=100, learning_rate=0.1, min_samples=5)
    global_model.fit(X_train_all, y_train_all)

    # Train per-home models
    home_models = {}
    homes_with_models = 0
    homes_with_fallback = 0

    test_homes = test_df["home_id"].unique()

    for home in tqdm(test_homes, desc="    Training per-home"):
        train_mask = train_df["home_id"] == home
        n_train = train_mask.sum()

        if n_train >= MIN_EPISODES_PER_HOME:
            X_home = train_df.loc[train_mask, feature_cols].fillna(0).values
            y_home = train_df.loc[train_mask, "log_duration"].values

            # Simpler model for per-home (less data)
            model = create_gbm_model(max_depth=3, n_estimators=50, learning_rate=0.1, min_samples=3)
            model.fit(X_home, y_home)
            home_models[home] = model
            homes_with_models += 1
        else:
            homes_with_fallback += 1

    # Predict
    predictions = np.zeros(len(test_df))

    for i, (idx, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        X = np.array([row[feature_cols].fillna(0).values])

        if home in home_models:
            log_pred = home_models[home].predict(X)[0]
        else:
            log_pred = global_model.predict(X)[0]

        predictions[i] = np.exp(log_pred)

    info = {
        "homes_with_models": homes_with_models,
        "homes_with_fallback": homes_with_fallback,
        "total_homes": len(test_homes),
        "pct_with_model": 100 * homes_with_models / len(test_homes),
    }

    print(f"    Homes with per-home model: {homes_with_models} ({info['pct_with_model']:.1f}%)")
    print(f"    Homes using global fallback: {homes_with_fallback}")

    return predictions, info


# =============================================================================
# Model 4: Hybrid - Global with per-home residual correction
# =============================================================================

def fit_hybrid_xgb_artifact(train_df: pd.DataFrame) -> dict:
    """
    Fit the reusable two-stage active time-to-target model:
    1. Global XGBoost for base prediction
    2. Per-home mean residual correction (shrunk toward 0)

    This is like a boosted version of the hierarchical model.
    """
    feature_cols = ALL_FEATURES

    # Stage 1: Global model
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["log_duration"].values

    global_model = create_gbm_model(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10)
    global_model.fit(X_train, y_train)

    # Compute residuals
    train_pred = global_model.predict(X_train)
    residuals = y_train - train_pred

    # Stage 2: Per-home residual correction with shrinkage
    train_df_with_resid = train_df.copy()
    train_df_with_resid["residual"] = residuals

    home_residuals = {}
    shrinkage_n = 10  # Shrink toward 0 for homes with few samples

    for home in train_df["home_id"].unique():
        mask = train_df["home_id"] == home
        if mask.sum() > 0:
            home_resid = residuals[mask].mean()
            n = mask.sum()
            # Shrink: weight = n / (n + shrinkage_n)
            weight = n / (n + shrinkage_n)
            home_residuals[home] = home_resid * weight

    # Also compute per-home×mode residuals
    home_mode_residuals = {}
    for home in train_df["home_id"].unique():
        for mode in [0, 1]:
            mask = (train_df["home_id"] == home) & (train_df["is_heating"] == mode)
            if mask.sum() >= 3:
                hm_resid = residuals[mask].mean()
                n = mask.sum()
                weight = n / (n + shrinkage_n)
                home_mode_residuals[(home, mode)] = hm_resid * weight

    return {
        "version": 1,
        "model_type": "hybrid_gbm_active_time_to_target",
        "backend": GBM_BACKEND,
        "model": global_model,
        "feature_cols": feature_cols,
        "predict_log": True,
        "home_residuals": home_residuals,
        "home_mode_residuals": home_mode_residuals,
        "residual_shrinkage_n": shrinkage_n,
        "min_duration": MIN_DURATION,
        "max_duration": MAX_DURATION,
        "n_train_rows": int(len(train_df)),
        "n_homes_with_correction": int(len(home_residuals)),
        "n_home_modes_with_correction": int(len(home_mode_residuals)),
    }


def predict_hybrid_xgb_artifact(artifact: dict, test_df: pd.DataFrame) -> np.ndarray:
    """Predict minutes from a fitted hybrid GBM artifact."""
    feature_cols = artifact["feature_cols"]
    global_model = artifact["model"]
    home_residuals = artifact["home_residuals"]
    home_mode_residuals = artifact["home_mode_residuals"]

    X_test = test_df[feature_cols].fillna(0).values
    global_pred = global_model.predict(X_test)

    # Predict
    predictions = np.zeros(len(test_df))

    for i, (idx, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        mode = int(row["is_heating"])

        log_pred = global_pred[i]

        # Add home×mode correction if available, else home correction
        if (home, mode) in home_mode_residuals:
            log_pred += home_mode_residuals[(home, mode)]
        elif home in home_residuals:
            log_pred += home_residuals[home]

        predictions[i] = np.exp(log_pred)

    return predictions


def save_hybrid_xgb_artifact(artifact: dict, output_path: Union[str, Path]) -> None:
    """Write a reusable model artifact for MPC active-time prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved active time-to-target model: {output_path}")


def create_drift_samples(
    drift_path: Union[str, Path] = DRIFT_DATA_PATH,
    max_rows_per_direction: int = 80_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create reusable passive-drift first-passage samples."""
    print("Creating drift first-passage samples...")
    t0 = time.time()
    samples = []

    for direction, drift_direction in [("heating", "warming_drift"), ("cooling", "cooling_drift")]:
        df = (
            pl.scan_parquet(drift_path)
            .filter(
                (pl.col("timestep_idx") == 0)
                & (pl.col("drift_direction") == drift_direction)
                & (pl.col("crossed_boundary") == True)
                & pl.col("time_to_boundary_min").is_between(MIN_DRIFT_DURATION, MAX_DRIFT_DURATION)
            )
            .select(
                "home_id",
                "timestamp",
                "start_temp",
                "boundary_temp",
                "Outdoor_Temperature",
                "time_to_boundary_min",
            )
            .collect()
            .to_pandas()
        )
        if len(df) > max_rows_per_direction:
            df = df.sample(n=max_rows_per_direction, random_state=random_seed)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["is_heating"] = 1 if direction == "heating" else 0
        if direction == "heating":
            df["margin"] = df["start_temp"] - df["boundary_temp"]
            df["signed_thermal_drive"] = df["start_temp"] - df["Outdoor_Temperature"]
        else:
            df["margin"] = df["boundary_temp"] - df["start_temp"]
            df["signed_thermal_drive"] = df["Outdoor_Temperature"] - df["start_temp"]

        df = df[df["margin"] > 1e-6].copy()
        df["log_margin"] = np.log(df["margin"] + 0.1)
        df["outdoor_temp"] = df["Outdoor_Temperature"]
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["duration_min"] = df["time_to_boundary_min"].astype(float)
        df["log_duration"] = np.log(df["duration_min"])
        samples.append(df)

    result = pd.concat(samples, ignore_index=True)
    print(f"  Drift samples: {len(result):,} rows in {time.time()-t0:.1f}s")
    return result


def fit_drift_xgb_artifact(drift_df: pd.DataFrame) -> dict:
    """Fit a reusable GBM passive-drift time-to-boundary model."""
    x_train = drift_df[DRIFT_FEATURES].fillna(0).values
    y_train = drift_df["log_duration"].values

    model = create_gbm_model(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10)
    model.fit(x_train, y_train)

    return {
        "version": 1,
        "model_type": "gbm_drift_time_to_boundary",
        "backend": GBM_BACKEND,
        "model": model,
        "feature_cols": DRIFT_FEATURES,
        "predict_log": True,
        "min_duration": MIN_DRIFT_DURATION,
        "max_duration": MAX_DRIFT_DURATION,
        "n_train_rows": int(len(drift_df)),
    }


def save_drift_xgb_artifact(artifact: dict, output_path: Union[str, Path]) -> None:
    """Write a reusable model artifact for MPC drift-time prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved drift time-to-boundary model: {output_path}")


def model_hybrid_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    return_artifact: bool = False,
) -> Union[Tuple[np.ndarray, dict], Tuple[np.ndarray, dict, dict]]:
    """Fit and evaluate the reusable hybrid active time-to-target model."""
    print("  Fitting hybrid XGBoost (global + per-home residual)...")

    artifact = fit_hybrid_xgb_artifact(train_df)
    predictions = predict_hybrid_xgb_artifact(artifact, test_df)

    # Feature importance from global model
    importance = get_feature_importance(artifact["model"], artifact["feature_cols"])
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {
        "n_homes_with_correction": artifact["n_homes_with_correction"],
        "n_home_modes_with_correction": artifact["n_home_modes_with_correction"],
        "residual_std": np.std(list(artifact["home_residuals"].values())),
        "top_features": top_features,
        "artifact_backend": artifact["backend"],
        "artifact_rows": artifact["n_train_rows"],
    }

    print(f"    Homes with correction: {info['n_homes_with_correction']}")
    print(f"    Residual std: {info['residual_std']:.3f}")

    if return_artifact:
        return predictions, info, artifact
    return predictions, info


# =============================================================================
# Reporting
# =============================================================================

def print_results(name: str, metrics: Metrics, info: dict = None):
    """Print results for a model."""
    print(f"\n  {name}")
    print(f"    Overall: MAE={metrics.mae:.1f} min, RMSE={metrics.rmse:.1f}, "
          f"Median={metrics.median_ae:.1f}, P90={metrics.p90:.1f}, Bias={metrics.bias:+.1f}")

    # By type
    print(f"    By type:")
    for ctype in ["heat_increase", "cool_decrease"]:
        if ctype in metrics.by_type:
            m = metrics.by_type[ctype]
            print(f"      {ctype}: MAE={m.mae:.1f}, Bias={m.bias:+.1f} (n={m.n_samples:,})")

    # By gap
    print(f"    By gap:")
    for gap_name in ["1-2°F", "2-3°F", "3-5°F", ">5°F"]:
        if gap_name in metrics.by_gap:
            m = metrics.by_gap[gap_name]
            print(f"      {gap_name}: MAE={m.mae:.1f} (n={m.n_samples:,})")

    # By system running
    if metrics.by_system:
        print(f"    By system state:")
        for state in ["running", "not_running"]:
            if state in metrics.by_system:
                m = metrics.by_system[state]
                print(f"      {state}: MAE={m.mae:.1f}, Bias={m.bias:+.1f} (n={m.n_samples:,})")


def main(
    model_output: Optional[Union[str, Path]] = DEFAULT_MODEL_OUTPUT,
    drift_model_output: Optional[Union[str, Path]] = DEFAULT_DRIFT_MODEL_OUTPUT,
):
    print("=" * 70)
    print("XGBoost Baselines for Time-to-Target Prediction")
    print("=" * 70)
    print()

    # Load and prepare data
    df = load_data()
    train_df, test_df = create_episode_samples(df, train_frac=0.7)

    print()
    print("Running XGBoost models...")
    print("-" * 70)

    results = []

    # Model 1: Global XGBoost (no home info)
    print("\n[1/5] Global XGBoost (no home info)")
    pred, info = model_global_xgb(train_df, test_df, predict_log=True)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values
    )
    results.append(("Global XGB", metrics, info))
    print_results("Global XGB", metrics, info)

    # Model 2: Global XGBoost (linear target)
    print("\n[2/5] Global XGBoost (linear target)")
    pred, info = model_global_xgb(train_df, test_df, predict_log=False)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values
    )
    results.append(("Global XGB (linear)", metrics, info))
    print_results("Global XGB (linear)", metrics, info)

    # Model 3: Global XGBoost with home encoding
    print("\n[3/5] Global XGBoost + Home Target Encoding")
    pred, info = model_global_xgb_home_encoded(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values
    )
    results.append(("Global XGB + Home Enc", metrics, info))
    print_results("Global XGB + Home Enc", metrics, info)

    # Model 4: Per-Home XGBoost
    print("\n[4/5] Per-Home XGBoost")
    pred, info = model_per_home_xgb(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values
    )
    results.append(("Per-Home XGB", metrics, info))
    print_results("Per-Home XGB", metrics, info)

    # Model 5: Hybrid (Global + per-home residual)
    print("\n[5/5] Hybrid XGBoost (Global + Per-Home Residual)")
    pred, info, artifact = model_hybrid_xgb(train_df, test_df, return_artifact=True)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values
    )
    results.append(("Hybrid XGB", metrics, info))
    print_results("Hybrid XGB", metrics, info)

    if model_output:
        save_hybrid_xgb_artifact(artifact, model_output)

    if drift_model_output:
        drift_df = create_drift_samples(DRIFT_DATA_PATH)
        drift_artifact = fit_drift_xgb_artifact(drift_df)
        save_drift_xgb_artifact(drift_artifact, drift_model_output)

    # Summary table
    print()
    print("=" * 85)
    print("SUMMARY: XGBoost Models (minutes)")
    print("=" * 85)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'Median':>8} {'P90':>8} {'Bias':>8}")
    print("-" * 85)
    for name, m, _ in results:
        print(f"{name:<25} {m.mae:>8.1f} {m.rmse:>8.1f} {m.median_ae:>8.1f} {m.p90:>8.1f} {m.bias:>+8.1f}")

    # Compare to best linear model (20.2 min from Hierarchical+Enhanced)
    print()
    print("Comparison to best linear model (Hierarchical+Enhanced: 20.2 min MAE):")
    baseline_mae = 20.2
    for name, m, _ in results:
        pct = 100 * (baseline_mae - m.mae) / baseline_mae
        direction = "better" if pct > 0 else "worse"
        print(f"  {name}: {abs(pct):.1f}% {direction} (MAE: {m.mae:.1f} min)")

    print()
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-output",
        default=str(DEFAULT_MODEL_OUTPUT),
        help="Path for the reusable active time-to-target model artifact. Use '' to skip saving.",
    )
    parser.add_argument(
        "--drift-model-output",
        default=str(DEFAULT_DRIFT_MODEL_OUTPUT),
        help="Path for the reusable drift time-to-boundary model artifact. Use '' to skip saving.",
    )
    args = parser.parse_args()
    main(
        model_output=args.model_output or None,
        drift_model_output=args.drift_model_output or None,
    )
