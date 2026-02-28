"""
Improved time-to-target prediction models.

Implements:
1. Log-duration transform + per-home+mode
2. Hierarchical (mixed effects) log-duration model
3. Enhanced features: thermal drive, seasonality, runtime

Usage: python scripts/run_improved_baselines.py
"""

import polars as pl
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from dataclasses import dataclass, field
from typing import Dict, Optional
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Config
DATA_PATH = "data/setpoint_responses.parquet"
MIN_DURATION = 5    # Minimum duration in minutes (1 timestep)
MAX_DURATION = 240  # Maximum duration in minutes (4 hours) - filter anomalies


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


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    change_types: np.ndarray,
    initial_gaps: np.ndarray,
) -> Metrics:
    """Compute all evaluation metrics."""
    errors = pred - target
    abs_errors = np.abs(errors)

    # Avoid division by zero for MAPE
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

    Each episode becomes one sample, predicting total duration from start conditions.
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

        # Shuffle and split episodes
        np.random.shuffle(episodes)
        n_train = max(1, int(len(episodes) * train_frac))
        train_episodes = set(episodes[:n_train])

        for ep_id in episodes:
            ep = home_df.filter(pl.col("episode_id") == ep_id).sort("timestep_idx")

            if len(ep) < 1:
                continue

            n_steps = len(ep)
            duration_min = n_steps * 5

            # Get start-of-episode features
            first_row = ep[0]

            # Extract values
            start_temp = first_row["Indoor_AverageTemperature"].item()
            target_sp = first_row["target_setpoint"].item()
            outdoor_temp = first_row["Outdoor_Temperature"].item()
            initial_gap = first_row["initial_gap"].item()
            change_type = first_row["change_type"].item()
            state = first_row["state"].item()
            timestamp = first_row["timestamp"].item()

            # Runtime at episode start (indicates if system was already running)
            heat_runtime = first_row["HeatingEquipmentStage1_RunTime"].item()
            cool_runtime = first_row["CoolingEquipmentStage1_RunTime"].item()
            fan_runtime = first_row["Fan_RunTime"].item()

            # Humidity
            indoor_humidity = first_row["Indoor_Humidity"].item()
            outdoor_humidity = first_row["Outdoor_Humidity"].item()

            # Skip if missing critical data
            if pd.isna(start_temp) or pd.isna(outdoor_temp) or pd.isna(initial_gap):
                continue
            if duration_min < MIN_DURATION or duration_min > MAX_DURATION:
                continue

            # Derived features
            thermal_drive = outdoor_temp - start_temp  # Positive = outdoor warmer

            # Signed thermal drive relative to goal
            if change_type == "heat_increase":
                # Heating: negative drive means outdoor is colder, makes heating harder
                signed_thermal_drive = thermal_drive  # Negative = harder
            else:
                # Cooling: positive drive means outdoor is warmer, makes cooling harder
                signed_thermal_drive = -thermal_drive  # Positive outdoor = harder (negative signed)

            # Time features from timestamp
            hour = timestamp.hour
            month = timestamp.month
            day_of_year = timestamp.timetuple().tm_yday

            # Cyclic encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            # Runtime features (fill NaN with 0)
            heat_runtime = 0 if pd.isna(heat_runtime) else heat_runtime
            cool_runtime = 0 if pd.isna(cool_runtime) else cool_runtime
            fan_runtime = 0 if pd.isna(fan_runtime) else fan_runtime

            # System was already running at episode start
            system_running = 1 if (heat_runtime > 0 or cool_runtime > 0) else 0

            # Compute cumulative runtime over first few steps (if available)
            # This captures "recent activity" context
            early_steps = min(3, len(ep))  # First 15 min or less
            early_heat_runtime = ep["HeatingEquipmentStage1_RunTime"][:early_steps].fill_null(0).sum()
            early_cool_runtime = ep["CoolingEquipmentStage1_RunTime"][:early_steps].fill_null(0).sum()

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
                "log_gap": np.log(np.abs(initial_gap) + 0.1),  # +0.1 to handle small gaps
                "start_temp": start_temp,
                "target_setpoint": target_sp,
                "outdoor_temp": outdoor_temp,
                "thermal_drive": thermal_drive,
                "signed_thermal_drive": signed_thermal_drive,
                # Time features
                "hour": hour,
                "month": month,
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
                "early_heat_runtime": early_heat_runtime,
                "early_cool_runtime": early_cool_runtime,
                # Mode indicator
                "is_heating": 1 if change_type == "heat_increase" else 0,
            })

    # Convert to DataFrame
    all_df = pd.DataFrame(records)
    train_df = all_df[all_df["is_train"]].copy()
    test_df = all_df[~all_df["is_train"]].copy()

    print(f"  Train: {len(train_df):,} episodes")
    print(f"  Test: {len(test_df):,} episodes")
    print(f"  Homes in both: {len(set(train_df['home_id']) & set(test_df['home_id']))}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Stats
    print(f"\n  Duration stats:")
    print(f"    Train - Mean: {train_df['duration_min'].mean():.1f} min, Median: {train_df['duration_min'].median():.1f} min")
    print(f"    Test  - Mean: {test_df['duration_min'].mean():.1f} min, Median: {test_df['duration_min'].median():.1f} min")

    return train_df, test_df


# =============================================================================
# Model 1: Log-Duration Per-Home + Mode (Baseline Upgrade)
# =============================================================================

def model_log_per_home_mode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Per-home + mode model working in log-duration space.

    log(duration) = log(k_home_mode) + log(gap)
    => duration = k_home_mode * gap

    But fitting in log-space handles multiplicative errors better.
    """
    print("  Fitting log-duration per-home+mode model...")

    # Target: log(duration)
    # Feature: log(gap)
    # Model: log(duration) = a + b * log(gap), per home+mode

    predictions = np.zeros(len(test_df))

    # Global fallback parameters
    global_params = {}
    for ctype in ["heat_increase", "cool_decrease"]:
        mask = train_df["change_type"] == ctype
        if mask.sum() > 0:
            X = train_df.loc[mask, "log_gap"].values.reshape(-1, 1)
            y = train_df.loc[mask, "log_duration"].values
            model = Ridge(alpha=0.1)
            model.fit(X, y)
            global_params[ctype] = {"intercept": model.intercept_, "slope": model.coef_[0]}

    # Per-home+mode parameters
    home_mode_params = {}
    test_homes = test_df["home_id"].unique()

    for home in test_homes:
        for ctype in ["heat_increase", "cool_decrease"]:
            train_mask = (train_df["home_id"] == home) & (train_df["change_type"] == ctype)

            if train_mask.sum() >= 5:
                X = train_df.loc[train_mask, "log_gap"].values.reshape(-1, 1)
                y = train_df.loc[train_mask, "log_duration"].values
                model = Ridge(alpha=0.1)
                model.fit(X, y)
                home_mode_params[(home, ctype)] = {"intercept": model.intercept_, "slope": model.coef_[0]}
            else:
                # Fall back to global
                home_mode_params[(home, ctype)] = global_params.get(ctype, {"intercept": 3.5, "slope": 1.0})

    # Predict
    for idx, row in test_df.iterrows():
        home = row["home_id"]
        ctype = row["change_type"]
        log_gap = row["log_gap"]

        params = home_mode_params.get((home, ctype), global_params.get(ctype, {"intercept": 3.5, "slope": 1.0}))
        log_pred = params["intercept"] + params["slope"] * log_gap
        predictions[test_df.index.get_loc(idx)] = np.exp(log_pred)

    # Collect learned k values (minutes per °F at gap=1)
    k_values = []
    for (home, ctype), params in home_mode_params.items():
        # At gap=1, log_gap=0, so log_duration = intercept
        # k = exp(intercept) is the time for 1°F gap
        k = np.exp(params["intercept"])
        k_values.append(k)

    info = {
        "k_min": min(k_values),
        "k_max": max(k_values),
        "k_median": np.median(k_values),
        "n_home_modes": len(home_mode_params),
    }

    return predictions, info


# =============================================================================
# Model 2: Hierarchical Mixed Effects Log-Duration
# =============================================================================

def model_hierarchical_log(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Hierarchical (mixed effects) model with partial pooling.

    log(duration) = fixed_effects + random_home_intercept + random_home_mode_slope

    Uses statsmodels MixedLM.
    """
    print("  Fitting hierarchical mixed effects model...")

    try:
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        print("    statsmodels not available, falling back to Ridge approximation")
        return model_hierarchical_ridge_approx(train_df, test_df)

    # Prepare data
    train_data = train_df.copy()

    # Create home-mode group for random slope
    train_data["home_mode"] = train_data["home_id"] + "_" + train_data["change_type"]

    # Formula: log_duration ~ log_gap + is_heating + thermal_drive + (1|home_id)
    # We'll fit with home as random intercept

    try:
        # Model with random intercept per home
        model = smf.mixedlm(
            "log_duration ~ log_gap + is_heating + signed_thermal_drive",
            train_data,
            groups=train_data["home_id"],
            # re_formula="~is_heating"  # Random slope for heating mode
        )
        result = model.fit(method="powell", maxiter=100)

        # Get fixed effects
        fixed_effects = result.fe_params

        # Get random effects (per-home intercepts)
        random_effects = result.random_effects  # Dict: home_id -> array

        info = {
            "fixed_intercept": fixed_effects.get("Intercept", 0),
            "fixed_log_gap": fixed_effects.get("log_gap", 1),
            "fixed_is_heating": fixed_effects.get("is_heating", 0),
            "fixed_thermal": fixed_effects.get("signed_thermal_drive", 0),
            "n_homes": len(random_effects),
            "re_std": np.std([v[0] for v in random_effects.values()]) if random_effects else 0,
        }

        print(f"    Fixed effects: intercept={info['fixed_intercept']:.3f}, "
              f"log_gap={info['fixed_log_gap']:.3f}, is_heating={info['fixed_is_heating']:.3f}")
        print(f"    Random effects std: {info['re_std']:.3f}")

        # Predict
        predictions = np.zeros(len(test_df))

        for idx, row in test_df.iterrows():
            home = row["home_id"]

            # Fixed part
            log_pred = fixed_effects.get("Intercept", 0)
            log_pred += fixed_effects.get("log_gap", 0) * row["log_gap"]
            log_pred += fixed_effects.get("is_heating", 0) * row["is_heating"]
            log_pred += fixed_effects.get("signed_thermal_drive", 0) * row["signed_thermal_drive"]

            # Random part (home intercept)
            if home in random_effects:
                log_pred += random_effects[home][0]

            predictions[test_df.index.get_loc(idx)] = np.exp(log_pred)

        return predictions, info

    except Exception as e:
        print(f"    MixedLM failed: {e}")
        print("    Falling back to Ridge approximation...")
        return model_hierarchical_ridge_approx(train_df, test_df)


def model_hierarchical_ridge_approx(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Ridge regression approximation of hierarchical model.

    Two-stage approach:
    1. Fit global model on residuals
    2. Learn per-home offsets with shrinkage
    """
    # Stage 1: Global model
    feature_cols = ["log_gap", "is_heating", "signed_thermal_drive"]
    X_train = train_df[feature_cols].values
    y_train = train_df["log_duration"].values

    global_model = Ridge(alpha=1.0)
    global_model.fit(X_train, y_train)

    global_pred = global_model.predict(X_train)
    residuals = y_train - global_pred

    # Stage 2: Per-home intercept offsets (shrunk toward 0)
    home_offsets = {}
    shrinkage = 0.5  # Blend with global (higher = more shrinkage)

    for home in train_df["home_id"].unique():
        mask = train_df["home_id"] == home
        if mask.sum() >= 3:
            home_residual = residuals[mask].mean()
            # Shrink based on sample size
            n = mask.sum()
            weight = n / (n + 10)  # More samples = less shrinkage
            home_offsets[home] = home_residual * weight
        else:
            home_offsets[home] = 0.0

    # Predict
    X_test = test_df[feature_cols].values
    global_pred_test = global_model.predict(X_test)

    predictions = np.zeros(len(test_df))
    for i, (idx, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        offset = home_offsets.get(home, 0.0)
        log_pred = global_pred_test[i] + offset
        predictions[i] = np.exp(log_pred)

    info = {
        "global_coefs": dict(zip(feature_cols, global_model.coef_)),
        "global_intercept": global_model.intercept_,
        "n_homes": len(home_offsets),
        "offset_std": np.std(list(home_offsets.values())),
    }

    return predictions, info


# =============================================================================
# Model 3: Hierarchical + Enhanced Features
# =============================================================================

def model_hierarchical_enhanced(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Hierarchical model with enhanced features:
    - Thermal drive (outdoor - indoor)
    - Seasonality (month, hour cyclic)
    - Runtime at episode start
    - Humidity
    """
    print("  Fitting hierarchical model with enhanced features...")

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("    statsmodels not available, using Ridge approximation")
        return model_enhanced_ridge(train_df, test_df)

    # Prepare data - fill missing values
    train_data = train_df.copy()
    test_data = test_df.copy()

    # Fill NaN in humidity
    for col in ["indoor_humidity", "outdoor_humidity"]:
        train_data[col] = train_data[col].fillna(50)
        test_data[col] = test_data[col].fillna(50)

    try:
        # Full model with all features
        formula = (
            "log_duration ~ log_gap + is_heating + signed_thermal_drive + "
            "hour_sin + hour_cos + month_sin + month_cos + "
            "system_running + outdoor_humidity"
        )

        model = smf.mixedlm(
            formula,
            train_data,
            groups=train_data["home_id"],
        )
        result = model.fit(method="powell", maxiter=100)

        fixed_effects = result.fe_params
        random_effects = result.random_effects

        info = {
            "fixed_effects": fixed_effects.to_dict(),
            "n_homes": len(random_effects),
            "re_std": np.std([v[0] for v in random_effects.values()]) if random_effects else 0,
        }

        print(f"    Fixed effects:")
        for k, v in fixed_effects.items():
            if k != "Intercept":
                print(f"      {k}: {v:.4f}")

        # Predict
        predictions = np.zeros(len(test_data))

        for i, (idx, row) in enumerate(test_data.iterrows()):
            home = row["home_id"]

            # Fixed part
            log_pred = fixed_effects.get("Intercept", 0)
            for feat in ["log_gap", "is_heating", "signed_thermal_drive",
                        "hour_sin", "hour_cos", "month_sin", "month_cos",
                        "system_running", "outdoor_humidity"]:
                if feat in fixed_effects:
                    log_pred += fixed_effects[feat] * row[feat]

            # Random part
            if home in random_effects:
                log_pred += random_effects[home][0]

            predictions[i] = np.exp(log_pred)

        return predictions, info

    except Exception as e:
        print(f"    MixedLM failed: {e}")
        return model_enhanced_ridge(train_df, test_df)


def model_enhanced_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Ridge approximation with enhanced features."""

    feature_cols = [
        "log_gap", "is_heating", "signed_thermal_drive",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "system_running", "outdoor_humidity"
    ]

    # Prepare data
    train_data = train_df.copy()
    test_data = test_df.copy()

    for col in feature_cols:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)

    X_train = train_data[feature_cols].values
    y_train = train_data["log_duration"].values

    # Global model
    global_model = Ridge(alpha=1.0)
    global_model.fit(X_train, y_train)

    # Residuals for per-home offsets
    global_pred = global_model.predict(X_train)
    residuals = y_train - global_pred

    # Per-home + mode offsets
    home_mode_offsets = {}
    for home in train_data["home_id"].unique():
        for mode in [0, 1]:  # is_heating
            mask = (train_data["home_id"] == home) & (train_data["is_heating"] == mode)
            if mask.sum() >= 3:
                offset = residuals[mask].mean()
                n = mask.sum()
                weight = n / (n + 10)
                home_mode_offsets[(home, mode)] = offset * weight

    # Predict
    X_test = test_data[feature_cols].values
    global_pred_test = global_model.predict(X_test)

    predictions = np.zeros(len(test_data))
    for i, (idx, row) in enumerate(test_data.iterrows()):
        home = row["home_id"]
        mode = int(row["is_heating"])
        offset = home_mode_offsets.get((home, mode), 0.0)
        log_pred = global_pred_test[i] + offset
        predictions[i] = np.exp(log_pred)

    info = {
        "global_coefs": dict(zip(feature_cols, global_model.coef_)),
        "global_intercept": global_model.intercept_,
        "n_home_modes": len(home_mode_offsets),
        "offset_std": np.std(list(home_mode_offsets.values())) if home_mode_offsets else 0,
    }

    print(f"    Global coefficients:")
    for feat, coef in zip(feature_cols, global_model.coef_):
        print(f"      {feat}: {coef:.4f}")

    return predictions, info


# =============================================================================
# Baseline: Original Per-Home + Mode (for comparison)
# =============================================================================

def model_baseline_per_home_mode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Original per-home + mode baseline (linear scale)."""
    print("  Fitting baseline per-home+mode (linear scale)...")

    predictions = np.zeros(len(test_df))

    # Global fallback k
    k_global = {}
    for ctype in ["heat_increase", "cool_decrease"]:
        mask = train_df["change_type"] == ctype
        if mask.sum() > 0:
            gaps = train_df.loc[mask, "abs_gap"].values
            times = train_df.loc[mask, "duration_min"].values
            k_global[ctype] = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)

    # Per-home+mode k
    home_mode_k = {}
    for home in test_df["home_id"].unique():
        for ctype in ["heat_increase", "cool_decrease"]:
            train_mask = (train_df["home_id"] == home) & (train_df["change_type"] == ctype)

            if train_mask.sum() >= 5:
                gaps = train_df.loc[train_mask, "abs_gap"].values
                times = train_df.loc[train_mask, "duration_min"].values
                k = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)
                k = np.clip(k, 5, 120)
                home_mode_k[(home, ctype)] = k
            else:
                home_mode_k[(home, ctype)] = k_global.get(ctype, 30)

    # Predict
    for i, (idx, row) in enumerate(test_df.iterrows()):
        home = row["home_id"]
        ctype = row["change_type"]
        gap = row["abs_gap"]
        k = home_mode_k.get((home, ctype), k_global.get(ctype, 30))
        predictions[i] = k * gap

    k_values = list(home_mode_k.values())
    info = {
        "k_min": min(k_values),
        "k_max": max(k_values),
        "k_median": np.median(k_values),
    }

    return predictions, info


# =============================================================================
# Main
# =============================================================================

def print_results(name: str, metrics: Metrics, info: dict = None):
    """Print results for a model."""
    print(f"\n  {name}")
    print(f"    Overall: MAE={metrics.mae:.1f} min, RMSE={metrics.rmse:.1f}, "
          f"Median={metrics.median_ae:.1f}, P90={metrics.p90:.1f}, Bias={metrics.bias:+.1f}")

    if info:
        for k, v in info.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            elif isinstance(v, dict) and len(v) <= 10:
                pass  # Skip printing large dicts

    # By type
    print(f"    By type:")
    for ctype in ["heat_increase", "cool_decrease"]:
        if ctype in metrics.by_type:
            m = metrics.by_type[ctype]
            print(f"      {ctype}: MAE={m.mae:.1f}, RMSE={m.rmse:.1f} (n={m.n_samples:,})")

    # By gap
    print(f"    By gap:")
    for gap_name in ["1-2°F", "2-3°F", "3-5°F", ">5°F"]:
        if gap_name in metrics.by_gap:
            m = metrics.by_gap[gap_name]
            print(f"      {gap_name}: MAE={m.mae:.1f}, RMSE={m.rmse:.1f} (n={m.n_samples:,})")


def main():
    print("=" * 70)
    print("Improved Time-to-Target Prediction Models")
    print("=" * 70)
    print()

    # Load and prepare data
    df = load_data()
    train_df, test_df = create_episode_samples(df, train_frac=0.7)

    print()
    print("Running models...")
    print("-" * 70)

    results = []

    # Baseline
    print("\n[1/4] Baseline: Per-Home + Mode (linear scale)")
    pred, info = model_baseline_per_home_mode(train_df, test_df)
    metrics = compute_metrics(pred, test_df["duration_min"].values,
                             test_df["change_type"].values, test_df["initial_gap"].values)
    results.append(("Baseline (linear)", metrics, info))
    print_results("Baseline (linear)", metrics, info)

    # Model 1: Log-duration
    print("\n[2/4] Log-Duration Per-Home + Mode")
    pred, info = model_log_per_home_mode(train_df, test_df)
    metrics = compute_metrics(pred, test_df["duration_min"].values,
                             test_df["change_type"].values, test_df["initial_gap"].values)
    results.append(("Log-Duration", metrics, info))
    print_results("Log-Duration", metrics, info)

    # Model 2: Hierarchical
    print("\n[3/4] Hierarchical Mixed Effects")
    pred, info = model_hierarchical_log(train_df, test_df)
    metrics = compute_metrics(pred, test_df["duration_min"].values,
                             test_df["change_type"].values, test_df["initial_gap"].values)
    results.append(("Hierarchical", metrics, info))
    print_results("Hierarchical", metrics, info)

    # Model 3: Hierarchical + Enhanced
    print("\n[4/4] Hierarchical + Enhanced Features")
    pred, info = model_hierarchical_enhanced(train_df, test_df)
    metrics = compute_metrics(pred, test_df["duration_min"].values,
                             test_df["change_type"].values, test_df["initial_gap"].values)
    results.append(("Hierarchical+Enhanced", metrics, info))
    print_results("Hierarchical+Enhanced", metrics, info)

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY: Episode-Start Prediction (minutes)")
    print("=" * 80)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'Median':>8} {'P90':>8} {'Bias':>8}")
    print("-" * 80)
    for name, m, _ in results:
        print(f"{name:<25} {m.mae:>8.1f} {m.rmse:>8.1f} {m.median_ae:>8.1f} {m.p90:>8.1f} {m.bias:>+8.1f}")

    # Improvement over baseline
    baseline_mae = results[0][1].mae
    print()
    print("Improvement over baseline:")
    for name, m, _ in results[1:]:
        pct = 100 * (baseline_mae - m.mae) / baseline_mae
        print(f"  {name}: {pct:+.1f}% MAE reduction")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
