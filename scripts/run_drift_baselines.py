"""
Drift baselines for boundary-crossing prediction.

This script evaluates passive HVAC-off drift as a first-passage-time task:
predict minutes until indoor temperature reaches the relevant comfort boundary.

Targets:
  - cooling_drift: time until indoor temperature reaches the cooling boundary
  - warming_drift: time until indoor temperature reaches the heating boundary

Usage:
  python3 scripts/run_drift_baselines.py
  python3 scripts/run_drift_baselines.py --sample-episodes 20000
  python3 scripts/run_drift_baselines.py --sample-homes 100
"""

import argparse
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

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
DATA_PATH = "data/drift_episodes.parquet"
MIN_TIME_TO_BOUNDARY = 15
MAX_TIME_TO_BOUNDARY = 480


@dataclass
class StratMetrics:
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    n_samples: int = 0


@dataclass
class Metrics:
    mae: float = 0.0
    rmse: float = 0.0
    median_ae: float = 0.0
    p90: float = 0.0
    bias: float = 0.0
    by_direction: Dict[str, StratMetrics] = field(default_factory=dict)
    by_delta: Dict[str, StratMetrics] = field(default_factory=dict)
    by_boundary_gap: Dict[str, StratMetrics] = field(default_factory=dict)


def summarize_subset(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> StratMetrics:
    if mask.sum() == 0:
        return StratMetrics()
    errors = pred[mask] - target[mask]
    return StratMetrics(
        mae=np.abs(errors).mean(),
        rmse=np.sqrt((errors ** 2).mean()),
        bias=errors.mean(),
        n_samples=int(mask.sum()),
    )


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    directions: np.ndarray,
    initial_deltas: np.ndarray,
    boundary_gaps: np.ndarray,
) -> Metrics:
    """Compute evaluation metrics."""
    errors = pred - target
    abs_errors = np.abs(errors)

    m = Metrics()
    m.mae = abs_errors.mean()
    m.rmse = np.sqrt((errors ** 2).mean())
    m.median_ae = np.median(abs_errors)
    m.p90 = np.percentile(abs_errors, 90)
    m.bias = errors.mean()

    for direction in ["cooling_drift", "warming_drift"]:
        m.by_direction[direction] = summarize_subset(pred, target, directions == direction)

    delta_bins = [("3-5F", 3, 5), ("5-10F", 5, 10), ("10-20F", 10, 20), (">20F", 20, np.inf)]
    for name, lo, hi in delta_bins:
        mask = (np.abs(initial_deltas) >= lo) & (np.abs(initial_deltas) < hi)
        m.by_delta[name] = summarize_subset(pred, target, mask)

    gap_bins = [("1F", 0, 1.5), ("2F", 1.5, 2.5), ("3-4F", 2.5, 4.5), (">4F", 4.5, np.inf)]
    for name, lo, hi in gap_bins:
        mask = (boundary_gaps >= lo) & (boundary_gaps < hi)
        m.by_boundary_gap[name] = summarize_subset(pred, target, mask)

    return m


def load_data() -> pl.DataFrame:
    print("Loading drift episodes...")
    t0 = time.time()
    df = pl.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    required_cols = {
        "episode_id",
        "home_id",
        "state",
        "split",
        "timestamp",
        "timestep_idx",
        "drift_direction",
        "start_temp",
        "start_outdoor",
        "initial_delta",
        "target_boundary",
        "boundary_temp",
        "crossed_boundary",
        "crossing_timestep_idx",
        "time_to_boundary_min",
        "distance_to_boundary",
        "signed_boundary_gap",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"{DATA_PATH} is missing reformulated drift columns: {missing_str}. "
            "Re-run python3 scripts/extract_drift_episodes.py first."
        )

    return df


def sample_data(
    df: pl.DataFrame,
    sample_homes: Optional[int] = None,
    sample_episodes: Optional[int] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Optionally sample homes and/or episodes while preserving full episode rows."""
    if sample_homes is None and sample_episodes is None:
        return df

    sampled_df = df
    rng = np.random.default_rng(seed)

    if sample_homes is not None:
        homes = sampled_df["home_id"].unique().to_list()
        if sample_homes <= 0:
            raise ValueError("--sample-homes must be positive.")
        if sample_homes < len(homes):
            chosen_homes = rng.choice(homes, size=sample_homes, replace=False).tolist()
            sampled_df = sampled_df.filter(pl.col("home_id").is_in(chosen_homes))
        print(
            f"  Home sample: {min(sample_homes, len(homes)):,} / {len(homes):,} homes"
        )

    if sample_episodes is not None:
        if sample_episodes <= 0:
            raise ValueError("--sample-episodes must be positive.")
        episodes = (
            sampled_df.select(["episode_id", "home_id"])
            .unique()
            .to_pandas()
        )
        n_episodes = len(episodes)
        if sample_episodes < n_episodes:
            chosen_idx = rng.choice(n_episodes, size=sample_episodes, replace=False)
            chosen_eps = episodes.iloc[chosen_idx]["episode_id"].tolist()
            sampled_df = sampled_df.filter(pl.col("episode_id").is_in(chosen_eps))
        print(
            f"  Episode sample: {min(sample_episodes, n_episodes):,} / {n_episodes:,} episodes"
        )

    stats = sampled_df.select(
        [
            pl.len().alias("rows"),
            pl.col("home_id").n_unique().alias("homes"),
            pl.col("episode_id").n_unique().alias("episodes"),
        ]
    ).to_dicts()[0]
    print(
        f"  Sampled dataset: {stats['rows']:,} rows, {stats['homes']:,} homes, "
        f"{stats['episodes']:,} episodes"
    )
    return sampled_df


def create_episode_samples(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Create episode-level samples for time-to-boundary prediction.

    Uses the dataset's home-level split assignment:
    - `train` homes for model fitting
    - `test` homes for final evaluation
    - `val` homes are held out from both by default
    """
    print(f"Creating episode-level samples (dataset split, max {MAX_TIME_TO_BOUNDARY} min)...")
    t0 = time.time()

    df = df.filter(pl.col("crossed_boundary") == True)

    records = []
    trajectories = {}
    homes = df["home_id"].unique().to_list()

    for home in tqdm(homes, desc="  Processing homes"):
        home_df = df.filter(pl.col("home_id") == home)
        episodes = home_df["episode_id"].unique().to_list()

        if len(episodes) < 1:
            continue

        for ep_id in episodes:
            ep = home_df.filter(pl.col("episode_id") == ep_id).sort("timestep_idx")
            if len(ep) < 2:
                continue

            first_row = ep[0]

            time_to_boundary_min = first_row["time_to_boundary_min"].item()
            if (
                pd.isna(time_to_boundary_min)
                or time_to_boundary_min < MIN_TIME_TO_BOUNDARY
                or time_to_boundary_min > MAX_TIME_TO_BOUNDARY
            ):
                continue

            start_temp = first_row["start_temp"].item()
            outdoor_temp = first_row["Outdoor_Temperature"].item()
            initial_delta = first_row["initial_delta"].item()
            drift_direction = first_row["drift_direction"].item()
            state = first_row["state"].item()
            split = first_row["split"].item()
            start_outdoor = first_row["start_outdoor"].item()
            boundary_temp = first_row["boundary_temp"].item()
            target_boundary = first_row["target_boundary"].item()
            distance_to_boundary = first_row["distance_to_boundary"].item()
            signed_boundary_gap = first_row["signed_boundary_gap"].item()
            timestamp = first_row["timestamp"].item()

            indoor_humidity = first_row["Indoor_Humidity"].item() if "Indoor_Humidity" in ep.columns else np.nan
            outdoor_humidity = first_row["Outdoor_Humidity"].item() if "Outdoor_Humidity" in ep.columns else np.nan

            if pd.isna(start_temp) or pd.isna(outdoor_temp) or pd.isna(boundary_temp):
                continue

            hour = timestamp.hour
            month = timestamp.month
            day_of_week = timestamp.weekday()

            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            indoor_humidity = 50 if pd.isna(indoor_humidity) else indoor_humidity
            outdoor_humidity = 50 if pd.isna(outdoor_humidity) else outdoor_humidity

            is_train = split == "train"
            is_test = split == "test"

            records.append(
                {
                    "home_id": home,
                    "episode_id": ep_id,
                    "is_train": is_train,
                    "is_test": is_test,
                    "state": state,
                    "split": split,
                    "drift_direction": drift_direction,
                    "target_boundary": target_boundary,
                    "time_to_boundary_min": time_to_boundary_min,
                    "log_time_to_boundary": np.log(time_to_boundary_min),
                    "initial_delta": initial_delta,
                    "abs_delta": abs(initial_delta),
                    "log_abs_delta": np.log(abs(initial_delta) + 0.1),
                    "start_temp": start_temp,
                    "outdoor_temp": outdoor_temp,
                    "start_outdoor": start_outdoor,
                    "boundary_temp": boundary_temp,
                    "distance_to_boundary": distance_to_boundary,
                    "log_distance_to_boundary": np.log(distance_to_boundary + 0.1),
                    "signed_boundary_gap": signed_boundary_gap,
                    "indoor_humidity": indoor_humidity,
                    "outdoor_humidity": outdoor_humidity,
                    "hour": hour,
                    "month": month,
                    "day_of_week": day_of_week,
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "month_sin": month_sin,
                    "month_cos": month_cos,
                    "is_cooling_drift": 1 if drift_direction == "cooling_drift" else 0,
                    "recent_indoor_slope_15m": first_row["recent_indoor_slope_15m"].item()
                    if "recent_indoor_slope_15m" in ep.columns
                    else np.nan,
                    "recent_indoor_slope_30m": first_row["recent_indoor_slope_30m"].item()
                    if "recent_indoor_slope_30m" in ep.columns
                    else np.nan,
                    "recent_outdoor_slope_15m": first_row["recent_outdoor_slope_15m"].item()
                    if "recent_outdoor_slope_15m" in ep.columns
                    else np.nan,
                    "recent_outdoor_slope_30m": first_row["recent_outdoor_slope_30m"].item()
                    if "recent_outdoor_slope_30m" in ep.columns
                    else np.nan,
                    "time_since_hvac_off_min": first_row["time_since_hvac_off_min"].item()
                    if "time_since_hvac_off_min" in ep.columns
                    else np.nan,
                }
            )

            temps = ep["Indoor_AverageTemperature"].to_numpy().astype(float)
            times_min = ep["timestep_idx"].to_numpy().astype(float) * 5.0
            trajectories[ep_id] = {
                "temps": temps,
                "times_min": times_min,
                "outdoor": start_outdoor,
                "start_temp": start_temp,
                "boundary_temp": boundary_temp,
                "is_train": is_train,
            }

    all_df = pd.DataFrame(records)
    train_df = all_df[all_df["is_train"]].copy()
    test_df = all_df[all_df["is_test"]].copy()
    val_df = all_df[(~all_df["is_train"]) & (~all_df["is_test"])].copy()

    print(f"  Train: {len(train_df):,} episodes")
    print(f"  Test: {len(test_df):,} episodes")
    print(f"  Val held out: {len(val_df):,} episodes")
    print(
        f"  Time-to-boundary - Mean: {all_df['time_to_boundary_min'].mean():.1f} min, "
        f"Median: {all_df['time_to_boundary_min'].median():.1f} min"
    )
    print(
        f"  Distance-to-boundary - Mean: {all_df['distance_to_boundary'].mean():.2f} F, "
        f"Median: {all_df['distance_to_boundary'].median():.2f} F"
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    return train_df, test_df, trajectories


def fit_newton_k(temps: np.ndarray, times_min: np.ndarray, outdoor: float) -> float:
    """
    Fit Newton's law decay constant k from an HVAC-off trajectory.
    T(t) = T_outdoor + (T_start - T_outdoor) * exp(-k * t)
    """
    delta0 = temps[0] - outdoor
    if abs(delta0) < 1.0:
        return np.nan

    normalized = (temps - outdoor) / delta0
    valid = normalized > 0.01
    if valid.sum() < 2:
        return np.nan

    t_valid = times_min[valid]
    log_norm = np.log(normalized[valid])
    denom = np.sum(t_valid ** 2)
    if denom <= 0:
        return np.nan

    k = -np.sum(t_valid * log_norm) / denom
    if k <= 0 or k > 1.0:
        return np.nan
    return k


def predict_newton_time(k: float, start_temp: float, outdoor: float, boundary_temp: float) -> float:
    """Predict time to a comfort boundary under Newtonian drift."""
    delta0 = start_temp - outdoor
    if abs(delta0) < 0.5 or k <= 0:
        return np.nan

    ratio = (boundary_temp - outdoor) / delta0
    if ratio <= 0 or ratio >= 1:
        return np.nan

    return -np.log(ratio) / k


def model_global_newton(train_df: pd.DataFrame, test_df: pd.DataFrame, trajectories: dict) -> tuple:
    """Global Newton baseline: fit one k across all training homes."""
    print("  Fitting global Newton boundary-crossing model...")

    k_values = []
    for ep_id in train_df["episode_id"]:
        traj = trajectories.get(ep_id)
        if traj is None or not traj["is_train"]:
            continue
        k = fit_newton_k(traj["temps"], traj["times_min"], traj["outdoor"])
        if not np.isnan(k):
            k_values.append(k)

    if not k_values:
        raise ValueError("Unable to fit any Newton k values from training episodes.")

    k_global = float(np.median(k_values))
    fallback = float(train_df["time_to_boundary_min"].median())

    print(f"    Global k: {k_global:.5f} (1/min), tau={1 / k_global:.1f} min")
    print(f"    k range: [{np.percentile(k_values, 10):.5f}, {np.percentile(k_values, 90):.5f}]")

    predictions = np.zeros(len(test_df))
    n_fallback = 0

    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = predict_newton_time(
            k_global,
            row["start_temp"],
            row["start_outdoor"],
            row["boundary_temp"],
        )
        if np.isnan(pred):
            pred = fallback
            n_fallback += 1
        predictions[i] = max(5, pred)

    info = {"k_global": k_global, "n_k_fitted": len(k_values), "n_fallback": n_fallback}
    return predictions, info


def model_per_home_newton(train_df: pd.DataFrame, test_df: pd.DataFrame, trajectories: dict) -> tuple:
    """Per-home Newton baseline with global fallback."""
    print("  Fitting per-home Newton boundary-crossing model...")

    home_k_values = {}
    all_k = []

    for ep_id in train_df["episode_id"]:
        traj = trajectories.get(ep_id)
        if traj is None or not traj["is_train"]:
            continue
        home = train_df.loc[train_df["episode_id"] == ep_id, "home_id"].iloc[0]
        k = fit_newton_k(traj["temps"], traj["times_min"], traj["outdoor"])
        if not np.isnan(k):
            home_k_values.setdefault(home, []).append(k)
            all_k.append(k)

    if not all_k:
        raise ValueError("Unable to fit any Newton k values from training episodes.")

    k_global = float(np.median(all_k))
    home_k = {home: float(np.median(ks)) for home, ks in home_k_values.items() if len(ks) >= 3}
    fallback = float(train_df["time_to_boundary_min"].median())

    print(f"    Homes with per-home k: {len(home_k)}")
    print(f"    Global fallback k: {k_global:.5f}")

    predictions = np.zeros(len(test_df))
    n_home_k = 0
    n_global_k = 0
    n_fallback = 0

    for i, (_, row) in enumerate(test_df.iterrows()):
        k = home_k.get(row["home_id"], k_global)
        if row["home_id"] in home_k:
            n_home_k += 1
        else:
            n_global_k += 1

        pred = predict_newton_time(
            k,
            row["start_temp"],
            row["start_outdoor"],
            row["boundary_temp"],
        )
        if np.isnan(pred):
            pred = fallback
            n_fallback += 1
        predictions[i] = max(5, pred)

    info = {
        "n_home_k": len(home_k),
        "n_used_home": n_home_k,
        "n_used_global": n_global_k,
        "n_fallback": n_fallback,
    }
    print(f"    Test: {n_home_k} used home k, {n_global_k} used global k")

    return predictions, info


CORE_FEATURES = [
    "log_abs_delta",
    "abs_delta",
    "is_cooling_drift",
    "start_temp",
    "outdoor_temp",
    "boundary_temp",
    "distance_to_boundary",
    "log_distance_to_boundary",
    "signed_boundary_gap",
]

TREND_FEATURES = [
    "recent_indoor_slope_15m",
    "recent_indoor_slope_30m",
    "recent_outdoor_slope_15m",
    "recent_outdoor_slope_30m",
    "time_since_hvac_off_min",
]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "hour",
    "month",
    "day_of_week",
]

HUMIDITY_FEATURES = [
    "indoor_humidity",
    "outdoor_humidity",
]

ALL_FEATURES = CORE_FEATURES + TREND_FEATURES + TIME_FEATURES + HUMIDITY_FEATURES


def create_gbm_model(max_depth: int = 6, n_estimators: int = 200, learning_rate: float = 0.1, min_samples: int = 10):
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
    if GBM_BACKEND == "xgboost":
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
    return HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=n_estimators,
        min_samples_leaf=min_samples,
        random_state=42,
    )


def get_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_cols, model.feature_importances_))
    return {}


def model_global_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Global GBM with no home information."""
    print(f"  Fitting global GBM (backend={GBM_BACKEND})...")

    feature_cols = ALL_FEATURES
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df["log_time_to_boundary"].values

    model = create_gbm_model()
    model.fit(X_train, y_train)

    pred = np.exp(model.predict(X_test))
    importance = get_feature_importance(model, feature_cols)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {"top_features": top_features}
    if top_features:
        print(f"    Top features: {[f'{k}:{v:.3f}' for k, v in top_features]}")

    return pred, info


def model_global_gbm_home_encoded(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Global GBM with home target encoding."""
    print("  Fitting global GBM + home target encoding...")

    global_mean = train_df["log_time_to_boundary"].mean()
    smoothing = 10

    home_stats = train_df.groupby("home_id").agg({"log_time_to_boundary": ["mean", "count"]})
    home_stats.columns = ["home_mean", "home_count"]
    home_stats["home_encoded"] = (
        (home_stats["home_count"] * home_stats["home_mean"] + smoothing * global_mean)
        / (home_stats["home_count"] + smoothing)
    )

    train_data = train_df.copy()
    test_data = test_df.copy()
    train_data["home_encoded"] = train_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)
    test_data["home_encoded"] = test_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)

    feature_cols = ALL_FEATURES + ["home_encoded"]
    X_train = train_data[feature_cols].fillna(0).values
    X_test = test_data[feature_cols].fillna(0).values
    y_train = train_data["log_time_to_boundary"].values

    model = create_gbm_model()
    model.fit(X_train, y_train)

    pred = np.exp(model.predict(X_test))
    importance = get_feature_importance(model, feature_cols)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {
        "n_homes_encoded": len(home_stats),
        "home_encoded_importance": importance.get("home_encoded", 0),
        "top_features": top_features,
    }
    if importance:
        print(f"    Home encoding importance: {info['home_encoded_importance']:.3f}")
        print(f"    Top features: {[f'{k}:{v:.3f}' for k, v in top_features]}")

    return pred, info


def model_hybrid_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Global GBM plus per-home residual correction."""
    print("  Fitting hybrid GBM (global + per-home residual)...")

    feature_cols = ALL_FEATURES
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df["log_time_to_boundary"].values

    global_model = create_gbm_model()
    global_model.fit(X_train, y_train)

    train_pred = global_model.predict(X_train)
    residuals = y_train - train_pred

    shrinkage_n = 10
    home_residuals = {}
    for home in train_df["home_id"].unique():
        mask = train_df["home_id"].values == home
        if mask.sum() > 0:
            n = mask.sum()
            weight = n / (n + shrinkage_n)
            home_residuals[home] = residuals[mask].mean() * weight

    global_pred = global_model.predict(X_test)
    predictions = np.zeros(len(test_df))

    for i, (_, row) in enumerate(test_df.iterrows()):
        log_pred = global_pred[i]
        home = row["home_id"]
        if home in home_residuals:
            log_pred += home_residuals[home]
        predictions[i] = np.exp(log_pred)

    importance = get_feature_importance(global_model, feature_cols)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {
        "n_homes_with_correction": len(home_residuals),
        "residual_std": np.std(list(home_residuals.values())) if home_residuals else 0.0,
        "top_features": top_features,
    }
    print(f"    Homes with correction: {info['n_homes_with_correction']}")
    print(f"    Residual std: {info['residual_std']:.3f}")

    return predictions, info


def print_results(name: str, metrics: Metrics):
    print(f"\n  {name}")
    print(
        f"    Overall: MAE={metrics.mae:.1f} min, RMSE={metrics.rmse:.1f}, "
        f"Median={metrics.median_ae:.1f}, P90={metrics.p90:.1f}, Bias={metrics.bias:+.1f}"
    )

    print("    By direction:")
    for direction in ["cooling_drift", "warming_drift"]:
        m = metrics.by_direction[direction]
        if m.n_samples > 0:
            print(f"      {direction}: MAE={m.mae:.1f}, Bias={m.bias:+.1f} (n={m.n_samples:,})")

    print("    By initial delta:")
    for name in ["3-5F", "5-10F", "10-20F", ">20F"]:
        m = metrics.by_delta[name]
        if m.n_samples > 0:
            print(f"      {name}: MAE={m.mae:.1f} (n={m.n_samples:,})")

    print("    By distance to boundary:")
    for name in ["1F", "2F", "3-4F", ">4F"]:
        m = metrics.by_boundary_gap[name]
        if m.n_samples > 0:
            print(f"      {name}: MAE={m.mae:.1f} (n={m.n_samples:,})")


def parse_args():
    parser = argparse.ArgumentParser(description="Run drift boundary-crossing baselines.")
    parser.add_argument(
        "--sample-homes",
        type=int,
        default=None,
        help="Sample this many homes before building episode-level training data.",
    )
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=None,
        help="Sample this many episodes before building episode-level training data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Drift Baselines for Boundary-Crossing Prediction")
    print("=" * 70)
    print("Target: time until HVAC-off drift reaches the relevant comfort boundary")
    print()

    df = load_data()
    df = sample_data(
        df,
        sample_homes=args.sample_homes,
        sample_episodes=args.sample_episodes,
        seed=args.seed,
    )
    train_df, test_df, trajectories = create_episode_samples(df)

    print()
    print("Running drift models...")
    print("-" * 70)

    results = []
    target = test_df["time_to_boundary_min"].values
    directions = test_df["drift_direction"].values
    initial_deltas = test_df["initial_delta"].values
    boundary_gaps = test_df["distance_to_boundary"].values

    print("\n[1/5] Global Newton's Law (median k)")
    pred, info = model_global_newton(train_df, test_df, trajectories)
    metrics = compute_metrics(pred, target, directions, initial_deltas, boundary_gaps)
    results.append(("Global Newton k", metrics, info))
    print_results("Global Newton k", metrics)

    print("\n[2/5] Per-Home Newton's Law")
    pred, info = model_per_home_newton(train_df, test_df, trajectories)
    metrics = compute_metrics(pred, target, directions, initial_deltas, boundary_gaps)
    results.append(("Per-Home Newton k", metrics, info))
    print_results("Per-Home Newton k", metrics)

    print("\n[3/5] Global GBM (no home info)")
    pred, info = model_global_gbm(train_df, test_df)
    metrics = compute_metrics(pred, target, directions, initial_deltas, boundary_gaps)
    results.append(("Global GBM", metrics, info))
    print_results("Global GBM", metrics)

    print("\n[4/5] Global GBM + Home Encoding")
    pred, info = model_global_gbm_home_encoded(train_df, test_df)
    metrics = compute_metrics(pred, target, directions, initial_deltas, boundary_gaps)
    results.append(("GBM + Home Enc", metrics, info))
    print_results("GBM + Home Enc", metrics)

    print("\n[5/5] Hybrid GBM (Global + Per-Home Residual)")
    pred, info = model_hybrid_gbm(train_df, test_df)
    metrics = compute_metrics(pred, target, directions, initial_deltas, boundary_gaps)
    results.append(("Hybrid GBM", metrics, info))
    print_results("Hybrid GBM", metrics)

    print()
    print("=" * 85)
    print("SUMMARY: Drift Boundary-Crossing Models (minutes)")
    print("=" * 85)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'Median':>8} {'P90':>8} {'Bias':>8}")
    print("-" * 85)
    for name, metrics, _ in results:
        print(
            f"{name:<25} {metrics.mae:>8.1f} {metrics.rmse:>8.1f} "
            f"{metrics.median_ae:>8.1f} {metrics.p90:>8.1f} {metrics.bias:>+8.1f}"
        )
    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
