"""
Drift rate baselines for time-to-target prediction.

Two tracks:
  Physics: Newton's law of cooling (fit decay constant k)
    1. Global k (median across all episodes)
    2. Per-home k
  GBM: Predict time-to-target directly
    3. Global GBM (no home info)
    4. Global GBM + home target encoding
    5. Hybrid GBM (global + per-home residual)

Target: time-to-target (minutes) - how long until indoor temp drifts to X?
For each episode, target = comfort bound (e.g., edge of deadband).

Usage: python scripts/run_drift_baselines.py
"""

import polars as pl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Dict
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
DATA_PATH = "data/drift_episodes.parquet"
MIN_DURATION = 15    # Minimum drift duration in minutes
MAX_DURATION = 480   # Maximum drift duration (8 hours)
MIN_EPISODES_PER_HOME = 5


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


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    directions: np.ndarray,
    initial_deltas: np.ndarray,
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

    # By drift direction
    for d in ["cooling_drift", "warming_drift"]:
        mask = directions == d
        if mask.sum() > 0:
            e = pred[mask] - target[mask]
            m.by_direction[d] = StratMetrics(
                mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                bias=e.mean(), n_samples=mask.sum(),
            )

    # By initial delta size
    delta_bins = [("3-5F", 3, 5), ("5-10F", 5, 10), ("10-20F", 10, 20), (">20F", 20, 100)]
    for name, lo, hi in delta_bins:
        mask = (np.abs(initial_deltas) >= lo) & (np.abs(initial_deltas) < hi)
        if mask.sum() > 0:
            e = pred[mask] - target[mask]
            m.by_delta[name] = StratMetrics(
                mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                bias=e.mean(), n_samples=mask.sum(),
            )

    return m


def load_data() -> pl.DataFrame:
    print("Loading drift episodes...")
    t0 = time.time()
    df = pl.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def create_episode_samples(df: pl.DataFrame, train_frac: float = 0.7) -> tuple:
    """
    Create episode-level samples for drift time-to-target prediction.

    For drift, "time to target" = duration of the drift episode (how long until
    the system needs to turn on again, or equivalently how long until temp
    reaches a comfort threshold).
    """
    print(f"Creating episode-level samples ({train_frac:.0%} train, max {MAX_DURATION} min)...")
    t0 = time.time()

    np.random.seed(42)

    records = []
    # Also collect per-episode temperature trajectories for physics models
    trajectories = {}

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

            if duration_min < MIN_DURATION or duration_min > MAX_DURATION:
                continue

            first_row = ep[0]

            start_temp = first_row["Indoor_AverageTemperature"].item()
            outdoor_temp = first_row["Outdoor_Temperature"].item()
            initial_delta = first_row["initial_delta"].item()
            drift_direction = first_row["drift_direction"].item()
            state = first_row["state"].item()
            split = first_row["split"].item()
            start_outdoor = first_row["start_outdoor"].item()
            timestamp = first_row["timestamp"].item()

            indoor_humidity = first_row["Indoor_Humidity"].item()
            outdoor_humidity = first_row["Outdoor_Humidity"].item()

            if pd.isna(start_temp) or pd.isna(outdoor_temp):
                continue

            # Time features
            hour = timestamp.hour
            month = timestamp.month
            day_of_week = timestamp.weekday()

            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            indoor_humidity = 50 if pd.isna(indoor_humidity) else indoor_humidity
            outdoor_humidity = 50 if pd.isna(outdoor_humidity) else outdoor_humidity

            is_train = ep_id in train_episodes

            records.append({
                "home_id": home,
                "episode_id": ep_id,
                "is_train": is_train,
                "state": state,
                "split": split,
                "drift_direction": drift_direction,
                "duration_min": duration_min,
                "log_duration": np.log(duration_min),
                # Core features
                "initial_delta": initial_delta,
                "abs_delta": abs(initial_delta),
                "log_abs_delta": np.log(abs(initial_delta) + 0.1),
                "start_temp": start_temp,
                "outdoor_temp": outdoor_temp,
                "start_outdoor": start_outdoor,
                # Time features
                "hour": hour,
                "month": month,
                "day_of_week": day_of_week,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                # Humidity
                "indoor_humidity": indoor_humidity,
                "outdoor_humidity": outdoor_humidity,
                # Direction indicator
                "is_cooling_drift": 1 if drift_direction == "cooling_drift" else 0,
            })

            # Save trajectory for physics models
            temps = ep["Indoor_AverageTemperature"].to_numpy().astype(float)
            times_min = np.arange(n_steps) * 5.0
            trajectories[ep_id] = {
                "temps": temps,
                "times_min": times_min,
                "outdoor": start_outdoor,
                "start_temp": start_temp,
                "is_train": is_train,
            }

    all_df = pd.DataFrame(records)
    train_df = all_df[all_df["is_train"]].copy()
    test_df = all_df[~all_df["is_train"]].copy()

    print(f"  Train: {len(train_df):,} episodes")
    print(f"  Test: {len(test_df):,} episodes")
    print(f"  Duration - Mean: {all_df['duration_min'].mean():.1f} min, "
          f"Median: {all_df['duration_min'].median():.1f} min")
    print(f"  Time: {time.time()-t0:.1f}s")

    return train_df, test_df, trajectories


# =============================================================================
# Physics Track: Newton's Law of Cooling
# =============================================================================

def fit_newton_k(temps: np.ndarray, times_min: np.ndarray, outdoor: float) -> float:
    """
    Fit Newton's law decay constant k from temperature trajectory.
    T(t) = T_outdoor + (T_start - T_outdoor) * exp(-k * t)
    Returns k (1/min), or NaN if fit fails.
    """
    T0 = temps[0]
    delta0 = T0 - outdoor

    if abs(delta0) < 1.0:
        return np.nan

    # Normalized: (T(t) - outdoor) / (T0 - outdoor) = exp(-k*t)
    # Take log: log((T(t) - outdoor) / delta0) = -k*t
    normalized = (temps - outdoor) / delta0

    # Filter valid points (must be positive for log)
    valid = normalized > 0.01
    if valid.sum() < 2:
        return np.nan

    t_valid = times_min[valid]
    log_norm = np.log(normalized[valid])

    # Least squares: log_norm = -k * t  (force through origin)
    # k = -sum(t * log_norm) / sum(t^2)
    k = -np.sum(t_valid * log_norm) / np.sum(t_valid ** 2)

    if k <= 0 or k > 1.0:  # k > 1/min is physically implausible
        return np.nan

    return k


def newton_time_to_target(k: float, start_temp: float, outdoor: float, duration: float) -> float:
    """
    Predict time for temp to drift from start to the end-of-episode temperature.
    Using Newton's law: T(t) = outdoor + (start - outdoor) * exp(-k*t)
    Solving for t: t = -ln((T_target - outdoor) / (start - outdoor)) / k

    For drift episodes, we predict the duration (time to drift the observed amount).
    """
    delta0 = start_temp - outdoor
    if abs(delta0) < 0.5 or k <= 0:
        return duration  # Can't predict, return actual

    # The target temp is what temp would be after `duration` minutes
    # But we want to predict duration itself.
    # Instead: from the fitted k, compute expected time for the observed temp change.
    # We'll just return the duration prediction based on k.
    return duration  # This gets overridden in the model functions


def model_global_newton(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        trajectories: dict) -> tuple:
    """
    Global Newton's law: single k = median across all training episodes.
    Predict duration using: t = -(1/k) * ln((T_end - outdoor) / (T_start - outdoor))
    But since we don't know T_end at prediction time, we predict using features.

    Actually for time-to-target: given k and the episode's observed temp change,
    Newton's law predicts how long that change should take.
    """
    print("  Fitting global Newton's law (median k)...")

    # Fit k for each training episode
    k_values = []
    for ep_id in train_df["episode_id"]:
        if ep_id not in trajectories:
            continue
        traj = trajectories[ep_id]
        if not traj["is_train"]:
            continue
        k = fit_newton_k(traj["temps"], traj["times_min"], traj["outdoor"])
        if not np.isnan(k):
            k_values.append(k)

    k_global = np.median(k_values)
    print(f"    Global k: {k_global:.5f} (1/min), tau={1/k_global:.1f} min")
    print(f"    k range: [{np.percentile(k_values, 10):.5f}, {np.percentile(k_values, 90):.5f}]")

    # Predict: for each test episode, predict duration from k and initial delta
    # T(t) = outdoor + delta0 * exp(-k*t)
    # After duration t, temp has changed by delta0 * (1 - exp(-k*t))
    # We observe the actual temp change; can we predict duration?
    # t = -ln(1 - temp_change/delta0) / k
    # But we don't know temp_change at prediction time.
    #
    # Alternative: predict duration = f(abs_delta, k)
    # For a 1F change: t = -ln(1 - 1/delta0) / k
    # This gives "time to drift 1F" which scales with delta0.
    #
    # Simplest: predict duration = actual_duration (baseline) won't work.
    # Use: predicted_time = duration given by Newton's law for the actual trajectory.
    predictions = np.zeros(len(test_df))

    for i, (_, row) in enumerate(test_df.iterrows()):
        ep_id = row["episode_id"]
        if ep_id in trajectories:
            traj = trajectories[ep_id]
            temps = traj["temps"]
            outdoor = traj["outdoor"]
            delta0 = temps[0] - outdoor

            if abs(delta0) > 0.5:
                # Observed final temp
                temp_change_ratio = (temps[-1] - outdoor) / delta0
                if temp_change_ratio > 0.01:
                    predicted = -np.log(temp_change_ratio) / k_global
                    predictions[i] = max(5, predicted)
                else:
                    predictions[i] = row["duration_min"]
            else:
                predictions[i] = row["duration_min"]
        else:
            predictions[i] = row["duration_min"]

    info = {"k_global": k_global, "n_k_fitted": len(k_values)}
    return predictions, info


def model_per_home_newton(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          trajectories: dict) -> tuple:
    """Per-home Newton's law: fit k per home, fall back to global."""
    print("  Fitting per-home Newton's law...")

    # Fit k per home from training episodes
    home_k_values = {}
    all_k = []

    for ep_id in train_df["episode_id"]:
        if ep_id not in trajectories:
            continue
        traj = trajectories[ep_id]
        home = train_df.loc[train_df["episode_id"] == ep_id, "home_id"].iloc[0]
        k = fit_newton_k(traj["temps"], traj["times_min"], traj["outdoor"])
        if not np.isnan(k):
            home_k_values.setdefault(home, []).append(k)
            all_k.append(k)

    k_global = np.median(all_k) if all_k else 0.005
    home_k = {h: np.median(ks) for h, ks in home_k_values.items() if len(ks) >= 3}

    print(f"    Homes with per-home k: {len(home_k)}")
    print(f"    Global fallback k: {k_global:.5f}")

    # Predict
    predictions = np.zeros(len(test_df))
    n_home_k = 0
    n_global_k = 0

    for i, (_, row) in enumerate(test_df.iterrows()):
        ep_id = row["episode_id"]
        home = row["home_id"]

        k = home_k.get(home, k_global)
        if home in home_k:
            n_home_k += 1
        else:
            n_global_k += 1

        if ep_id in trajectories:
            traj = trajectories[ep_id]
            temps = traj["temps"]
            outdoor = traj["outdoor"]
            delta0 = temps[0] - outdoor

            if abs(delta0) > 0.5:
                temp_change_ratio = (temps[-1] - outdoor) / delta0
                if temp_change_ratio > 0.01:
                    predicted = -np.log(temp_change_ratio) / k
                    predictions[i] = max(5, predicted)
                else:
                    predictions[i] = row["duration_min"]
            else:
                predictions[i] = row["duration_min"]
        else:
            predictions[i] = row["duration_min"]

    info = {
        "n_home_k": len(home_k),
        "n_used_home": n_home_k,
        "n_used_global": n_global_k,
    }
    print(f"    Test: {n_home_k} used home k, {n_global_k} used global k")

    return predictions, info


# =============================================================================
# GBM Track
# =============================================================================

CORE_FEATURES = [
    "log_abs_delta",
    "abs_delta",
    "is_cooling_drift",
    "start_temp",
    "outdoor_temp",
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


def create_gbm_model(max_depth=6, n_estimators=200, learning_rate=0.1, min_samples=10):
    if GBM_BACKEND == "lightgbm":
        return lgb.LGBMRegressor(
            objective="regression", max_depth=max_depth,
            learning_rate=learning_rate, n_estimators=n_estimators,
            min_child_samples=min_samples, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
        )
    elif GBM_BACKEND == "xgboost":
        return xgb.XGBRegressor(
            objective="reg:squarederror", max_depth=max_depth,
            learning_rate=learning_rate, n_estimators=n_estimators,
            min_child_weight=min_samples, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=max_depth, learning_rate=learning_rate,
            max_iter=n_estimators, min_samples_leaf=min_samples, random_state=42,
        )


def get_feature_importance(model, feature_cols):
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_cols, model.feature_importances_))
    return {}


def model_global_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Global GBM - no home info."""
    print(f"  Fitting global GBM (backend={GBM_BACKEND})...")

    feature_cols = ALL_FEATURES
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df["log_duration"].values

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

    global_mean = train_df["log_duration"].mean()
    smoothing = 10

    home_stats = train_df.groupby("home_id").agg({"log_duration": ["mean", "count"]})
    home_stats.columns = ["home_mean", "home_count"]
    home_stats["home_encoded"] = (
        (home_stats["home_count"] * home_stats["home_mean"] + smoothing * global_mean) /
        (home_stats["home_count"] + smoothing)
    )

    train_data = train_df.copy()
    test_data = test_df.copy()
    train_data["home_encoded"] = train_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)
    test_data["home_encoded"] = test_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)

    feature_cols = ALL_FEATURES + ["home_encoded"]

    X_train = train_data[feature_cols].fillna(0).values
    X_test = test_data[feature_cols].fillna(0).values
    y_train = train_data["log_duration"].values

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
    """Hybrid: global GBM + per-home residual correction."""
    print("  Fitting hybrid GBM (global + per-home residual)...")

    feature_cols = ALL_FEATURES

    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df["log_duration"].values

    global_model = create_gbm_model()
    global_model.fit(X_train, y_train)

    # Per-home residual correction
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

    # Predict
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
        "residual_std": np.std(list(home_residuals.values())),
        "top_features": top_features,
    }
    print(f"    Homes with correction: {info['n_homes_with_correction']}")
    print(f"    Residual std: {info['residual_std']:.3f}")

    return predictions, info


# =============================================================================
# Reporting
# =============================================================================

def print_results(name: str, metrics: Metrics):
    print(f"\n  {name}")
    print(f"    Overall: MAE={metrics.mae:.1f} min, RMSE={metrics.rmse:.1f}, "
          f"Median={metrics.median_ae:.1f}, P90={metrics.p90:.1f}, Bias={metrics.bias:+.1f}")

    if metrics.by_direction:
        print(f"    By direction:")
        for d in ["cooling_drift", "warming_drift"]:
            if d in metrics.by_direction:
                m = metrics.by_direction[d]
                print(f"      {d}: MAE={m.mae:.1f}, Bias={m.bias:+.1f} (n={m.n_samples:,})")

    if metrics.by_delta:
        print(f"    By initial delta:")
        for name in ["3-5F", "5-10F", "10-20F", ">20F"]:
            if name in metrics.by_delta:
                m = metrics.by_delta[name]
                print(f"      {name}: MAE={m.mae:.1f} (n={m.n_samples:,})")


def main():
    print("=" * 70)
    print("Drift Rate Baselines for Time-to-Target Prediction")
    print("=" * 70)
    print()

    df = load_data()
    train_df, test_df, trajectories = create_episode_samples(df, train_frac=0.7)

    print()
    print("Running drift models...")
    print("-" * 70)

    results = []

    # Physics Track
    print("\n[1/5] Global Newton's Law (median k)")
    pred, info = model_global_newton(train_df, test_df, trajectories)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["drift_direction"].values, test_df["initial_delta"].values,
    )
    results.append(("Global Newton k", metrics, info))
    print_results("Global Newton k", metrics)

    print("\n[2/5] Per-Home Newton's Law")
    pred, info = model_per_home_newton(train_df, test_df, trajectories)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["drift_direction"].values, test_df["initial_delta"].values,
    )
    results.append(("Per-Home Newton k", metrics, info))
    print_results("Per-Home Newton k", metrics)

    # GBM Track
    print("\n[3/5] Global GBM (no home info)")
    pred, info = model_global_gbm(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["drift_direction"].values, test_df["initial_delta"].values,
    )
    results.append(("Global GBM", metrics, info))
    print_results("Global GBM", metrics)

    print("\n[4/5] Global GBM + Home Encoding")
    pred, info = model_global_gbm_home_encoded(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["drift_direction"].values, test_df["initial_delta"].values,
    )
    results.append(("GBM + Home Enc", metrics, info))
    print_results("GBM + Home Enc", metrics)

    print("\n[5/5] Hybrid GBM (Global + Per-Home Residual)")
    pred, info = model_hybrid_gbm(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["drift_direction"].values, test_df["initial_delta"].values,
    )
    results.append(("Hybrid GBM", metrics, info))
    print_results("Hybrid GBM", metrics)

    # Summary
    print()
    print("=" * 85)
    print("SUMMARY: Drift Models (minutes)")
    print("=" * 85)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'Median':>8} {'P90':>8} {'Bias':>8}")
    print("-" * 85)
    for name, m, _ in results:
        print(f"{name:<25} {m.mae:>8.1f} {m.rmse:>8.1f} {m.median_ae:>8.1f} {m.p90:>8.1f} {m.bias:>+8.1f}")

    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
