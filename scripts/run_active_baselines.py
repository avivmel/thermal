"""
Active model baselines for time-to-target prediction.

Re-trains the active model on setpoint response episodes, keeping both
cold-start and already-running episodes (MPC needs both).
system_running is kept as a feature (highly predictive).

Model variants:
1. Global GBM (no home info)
2. Global GBM + home target encoding
3. Per-home GBM (with global fallback)
4. Hybrid GBM (global + per-home residual)

This parallels run_xgboost_baselines.py, confirming the active model
works standalone alongside the new drift model.

Usage: python scripts/run_active_baselines.py
"""

import polars as pl
import numpy as np
import pandas as pd
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
DATA_PATH = "data/setpoint_responses.parquet"
MIN_DURATION = 5
MAX_DURATION = 240  # 4 hours - filter anomalies
MIN_EPISODES_PER_HOME = 10


@dataclass
class StratMetrics:
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

    for ctype in ["heat_increase", "cool_decrease"]:
        mask = change_types == ctype
        if mask.sum() > 0:
            e = pred[mask] - target[mask]
            valid = target[mask] > 1
            m.by_type[ctype] = StratMetrics(
                mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                mape=100 * (np.abs(e[valid]) / target[mask][valid]).mean() if valid.sum() > 0 else 0,
                bias=e.mean(), n_samples=mask.sum(),
            )

    gap_bins = [("1-2F", 1, 2), ("2-3F", 2, 3), ("3-5F", 3, 5), (">5F", 5, 100)]
    for name, lo, hi in gap_bins:
        mask = (np.abs(initial_gaps) >= lo) & (np.abs(initial_gaps) < hi)
        if mask.sum() > 0:
            e = pred[mask] - target[mask]
            valid = target[mask] > 1
            m.by_gap[name] = StratMetrics(
                mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                mape=100 * (np.abs(e[valid]) / target[mask][valid]).mean() if valid.sum() > 0 else 0,
                bias=e.mean(), n_samples=mask.sum(),
            )

    if system_running is not None:
        for state, label in [(1, "running"), (0, "not_running")]:
            mask = system_running == state
            if mask.sum() > 0:
                e = pred[mask] - target[mask]
                m.by_system[label] = StratMetrics(
                    mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                    bias=e.mean(), n_samples=mask.sum(),
                )

    return m


def load_data() -> pl.DataFrame:
    print("Loading setpoint response data...")
    t0 = time.time()
    df = pl.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def create_episode_samples(df: pl.DataFrame, train_frac: float = 0.7) -> tuple:
    """
    Create episode-level samples for active time-to-target prediction.

    Keeps ALL episodes (cold-start + already-running) since MPC needs
    to predict from any starting state.
    """
    print(f"Creating episode-level samples ({train_frac:.0%} train, max {MAX_DURATION} min)...")
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
    print(f"  Homes: {all_df['home_id'].nunique()}")
    print(f"  System running: {(all_df['system_running'] == 1).sum():,} ({100*(all_df['system_running'] == 1).mean():.1f}%)")
    print(f"  Duration - Mean: {all_df['duration_min'].mean():.1f} min, Median: {all_df['duration_min'].median():.1f} min")
    print(f"  Time: {time.time()-t0:.1f}s")

    return train_df, test_df


# =============================================================================
# Features
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


# =============================================================================
# GBM Model Factory
# =============================================================================

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


# =============================================================================
# Model 1: Global GBM
# =============================================================================

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


# =============================================================================
# Model 2: Global GBM + Home Target Encoding
# =============================================================================

def model_global_gbm_home_encoded(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Global GBM with home target encoding (mean log-duration with shrinkage)."""
    print("  Fitting global GBM + home target encoding...")

    global_mean = train_df["log_duration"].mean()
    smoothing = 10

    home_stats = train_df.groupby("home_id").agg({"log_duration": ["mean", "count"]})
    home_stats.columns = ["home_mean", "home_count"]
    home_stats["home_encoded"] = (
        (home_stats["home_count"] * home_stats["home_mean"] + smoothing * global_mean) /
        (home_stats["home_count"] + smoothing)
    )

    # Home x mode encoding
    home_mode_stats = train_df.groupby(["home_id", "is_heating"]).agg({"log_duration": ["mean", "count"]})
    home_mode_stats.columns = ["hm_mean", "hm_count"]
    home_mode_stats["home_mode_encoded"] = (
        (home_mode_stats["hm_count"] * home_mode_stats["hm_mean"] + smoothing * global_mean) /
        (home_mode_stats["hm_count"] + smoothing)
    )

    train_data = train_df.copy()
    test_data = test_df.copy()

    train_data["home_encoded"] = train_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)
    test_data["home_encoded"] = test_data["home_id"].map(home_stats["home_encoded"]).fillna(global_mean)

    hm_encoding = home_mode_stats["home_mode_encoded"].to_dict()

    def get_hm_encoding(row):
        return hm_encoding.get((row["home_id"], row["is_heating"]), global_mean)

    train_data["home_mode_encoded"] = train_data.apply(get_hm_encoding, axis=1)
    test_data["home_mode_encoded"] = test_data.apply(get_hm_encoding, axis=1)

    feature_cols = ALL_FEATURES + ["home_encoded", "home_mode_encoded"]

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
        "home_mode_encoded_importance": importance.get("home_mode_encoded", 0),
        "top_features": top_features,
    }

    if importance:
        print(f"    Home encoding importance: {info['home_encoded_importance']:.3f}")
        print(f"    Home×mode encoding importance: {info['home_mode_encoded_importance']:.3f}")
        print(f"    Top features: {[f'{k}:{v:.3f}' for k, v in top_features]}")

    return pred, info


# =============================================================================
# Model 3: Per-Home GBM
# =============================================================================

def model_per_home_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Separate GBM per home, global fallback."""
    print(f"  Fitting per-home GBM (min {MIN_EPISODES_PER_HOME} episodes)...")

    feature_cols = ALL_FEATURES

    # Global fallback
    X_train_all = train_df[feature_cols].fillna(0).values
    y_train_all = train_df["log_duration"].values

    global_model = create_gbm_model(max_depth=4, n_estimators=100, min_samples=5)
    global_model.fit(X_train_all, y_train_all)

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

            model = create_gbm_model(max_depth=3, n_estimators=50, min_samples=3)
            model.fit(X_home, y_home)
            home_models[home] = model
            homes_with_models += 1
        else:
            homes_with_fallback += 1

    predictions = np.zeros(len(test_df))
    for i, (idx, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        X = np.array([row[feature_cols].fillna(0).values])
        if home in home_models:
            predictions[i] = np.exp(home_models[home].predict(X)[0])
        else:
            predictions[i] = np.exp(global_model.predict(X)[0])

    info = {
        "homes_with_models": homes_with_models,
        "homes_with_fallback": homes_with_fallback,
        "pct_with_model": 100 * homes_with_models / max(len(test_homes), 1),
    }
    print(f"    Homes with per-home model: {homes_with_models} ({info['pct_with_model']:.1f}%)")
    print(f"    Homes using global fallback: {homes_with_fallback}")

    return predictions, info


# =============================================================================
# Model 4: Hybrid GBM
# =============================================================================

def model_hybrid_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Global GBM + per-home residual correction (with shrinkage)."""
    print("  Fitting hybrid GBM (global + per-home residual)...")

    feature_cols = ALL_FEATURES

    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df["log_duration"].values

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

    # Per-home x mode residuals
    home_mode_residuals = {}
    for home in train_df["home_id"].unique():
        for mode in [0, 1]:
            mask = (train_df["home_id"].values == home) & (train_df["is_heating"].values == mode)
            if mask.sum() >= 3:
                n = mask.sum()
                weight = n / (n + shrinkage_n)
                home_mode_residuals[(home, mode)] = residuals[mask].mean() * weight

    global_pred = global_model.predict(X_test)
    predictions = np.zeros(len(test_df))

    for i, (idx, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        mode = int(row["is_heating"])
        log_pred = global_pred[i]

        if (home, mode) in home_mode_residuals:
            log_pred += home_mode_residuals[(home, mode)]
        elif home in home_residuals:
            log_pred += home_residuals[home]

        predictions[i] = np.exp(log_pred)

    importance = get_feature_importance(global_model, feature_cols)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5] if importance else []

    info = {
        "n_homes_with_correction": len(home_residuals),
        "n_home_modes_with_correction": len(home_mode_residuals),
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

    print(f"    By type:")
    for ctype in ["heat_increase", "cool_decrease"]:
        if ctype in metrics.by_type:
            m = metrics.by_type[ctype]
            print(f"      {ctype}: MAE={m.mae:.1f}, Bias={m.bias:+.1f} (n={m.n_samples:,})")

    print(f"    By gap:")
    for gap_name in ["1-2F", "2-3F", "3-5F", ">5F"]:
        if gap_name in metrics.by_gap:
            m = metrics.by_gap[gap_name]
            print(f"      {gap_name}: MAE={m.mae:.1f} (n={m.n_samples:,})")

    if metrics.by_system:
        print(f"    By system state:")
        for state in ["running", "not_running"]:
            if state in metrics.by_system:
                m = metrics.by_system[state]
                print(f"      {state}: MAE={m.mae:.1f}, Bias={m.bias:+.1f} (n={m.n_samples:,})")


def main():
    print("=" * 70)
    print("Active Model Baselines for Time-to-Target Prediction")
    print("=" * 70)
    print()

    df = load_data()
    train_df, test_df = create_episode_samples(df, train_frac=0.7)

    print()
    print("Running active models...")
    print("-" * 70)

    results = []

    # Model 1: Global GBM
    print("\n[1/4] Global GBM (no home info)")
    pred, info = model_global_gbm(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values,
    )
    results.append(("Global GBM", metrics, info))
    print_results("Global GBM", metrics)

    # Model 2: Global GBM + Home Encoding
    print("\n[2/4] Global GBM + Home Target Encoding")
    pred, info = model_global_gbm_home_encoded(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values,
    )
    results.append(("GBM + Home Enc", metrics, info))
    print_results("GBM + Home Enc", metrics)

    # Model 3: Per-Home GBM
    print("\n[3/4] Per-Home GBM")
    pred, info = model_per_home_gbm(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values,
    )
    results.append(("Per-Home GBM", metrics, info))
    print_results("Per-Home GBM", metrics)

    # Model 4: Hybrid GBM
    print("\n[4/4] Hybrid GBM (Global + Per-Home Residual)")
    pred, info = model_hybrid_gbm(train_df, test_df)
    metrics = compute_metrics(
        pred, test_df["duration_min"].values,
        test_df["change_type"].values, test_df["initial_gap"].values,
        test_df["system_running"].values,
    )
    results.append(("Hybrid GBM", metrics, info))
    print_results("Hybrid GBM", metrics)

    # Summary
    print()
    print("=" * 85)
    print("SUMMARY: Active Models (minutes)")
    print("=" * 85)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'Median':>8} {'P90':>8} {'Bias':>8}")
    print("-" * 85)
    for name, m, _ in results:
        print(f"{name:<25} {m.mae:>8.1f} {m.rmse:>8.1f} {m.median_ae:>8.1f} {m.p90:>8.1f} {m.bias:>+8.1f}")

    # Compare to previous best (Hybrid XGB: 18.7 min MAE from run_xgboost_baselines.py)
    print()
    print("Comparison to previous best (Hybrid XGB: 18.7 min MAE):")
    baseline_mae = 18.7
    for name, m, _ in results:
        pct = 100 * (baseline_mae - m.mae) / baseline_mae
        direction = "better" if pct > 0 else "worse"
        print(f"  {name}: {abs(pct):.1f}% {direction} (MAE: {m.mae:.1f} min)")

    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
