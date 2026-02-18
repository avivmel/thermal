"""
Per-home baseline experiment.

Trains separate models for each home with temporal train/test split.
This measures the "oracle" upper bound - what linear models could achieve
with perfect home-specific knowledge.

Usage: python scripts/run_per_home_baselines.py
"""

import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

# Config
DATA_PATH = "data/thermal_dataset.csv"
HORIZON = 6  # 30 minutes ahead
SAMPLE_RATE = 12  # Sample every 12 timesteps (1 hour) - denser for per-home
TRAIN_RATIO = 0.8  # 80% train, 20% test (temporal split)
MIN_SAMPLES = 100  # Minimum samples per home to include


@dataclass
class HomeResult:
    """Results for a single home."""
    home_id: str
    state: str
    n_train: int
    n_test: int
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    targets: np.ndarray = None
    modes: np.ndarray = None


def load_data() -> pl.DataFrame:
    """Load data with Polars."""
    print("Loading data...")
    t0 = time.time()
    df = pl.scan_csv(DATA_PATH).collect()
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def create_home_samples(home_df: pl.DataFrame) -> Tuple[dict, dict]:
    """Create train/test samples for a single home with temporal split."""
    home_df = home_df.sort("timestamp")

    indoor = home_df["Indoor_AverageTemperature"].to_numpy()
    outdoor = home_df["Outdoor_Temperature"].to_numpy()
    heat = home_df["Indoor_HeatSetpoint"].to_numpy()
    cool = home_df["Indoor_CoolSetpoint"].to_numpy()
    timestamps = home_df["timestamp"].to_list()

    n = len(indoor)
    valid_end = n - HORIZON

    if valid_end <= 0:
        return None, None

    # Create all valid samples
    all_samples = {
        "indoor_temp": [],
        "outdoor_temp": [],
        "outdoor_temp_future": [],
        "heat_sp": [],
        "cool_sp": [],
        "target": [],
        "hour": [],
        "thermal_drive": [],
        "is_heating": [],
        "is_cooling": [],
        "gap_to_target": [],
        "mode": [],
    }

    indices = np.arange(0, valid_end, SAMPLE_RATE)

    for t in indices:
        curr_indoor = indoor[t]
        fut_indoor = indoor[t + HORIZON]
        curr_outdoor = outdoor[t]
        fut_outdoor = outdoor[t + HORIZON]
        curr_heat = heat[t]
        curr_cool = cool[t]

        # Skip invalid
        if np.isnan(curr_indoor) or curr_indoor == 0:
            continue
        if np.isnan(fut_indoor) or fut_indoor == 0:
            continue
        if np.isnan(curr_outdoor) or curr_outdoor == 0:
            continue
        if np.isnan(fut_outdoor) or fut_outdoor == 0:
            continue
        if np.isnan(curr_heat) or np.isnan(curr_cool):
            continue

        # Determine mode and gap
        if curr_indoor < curr_heat:
            mode = "heating"
            is_heating = 1
            is_cooling = 0
            gap = curr_heat - curr_indoor
        elif curr_indoor > curr_cool:
            mode = "cooling"
            is_heating = 0
            is_cooling = 1
            gap = curr_indoor - curr_cool
        else:
            mode = "passive"
            is_heating = 0
            is_cooling = 0
            gap = 0

        # Extract hour
        try:
            hour = int(timestamps[t].split(" ")[1].split(":")[0])
        except:
            hour = 12

        all_samples["indoor_temp"].append(curr_indoor)
        all_samples["outdoor_temp"].append(curr_outdoor)
        all_samples["outdoor_temp_future"].append(fut_outdoor)
        all_samples["heat_sp"].append(curr_heat)
        all_samples["cool_sp"].append(curr_cool)
        all_samples["target"].append(fut_indoor)
        all_samples["hour"].append(hour)
        all_samples["thermal_drive"].append(curr_outdoor - curr_indoor)
        all_samples["is_heating"].append(is_heating)
        all_samples["is_cooling"].append(is_cooling)
        all_samples["gap_to_target"].append(gap)
        all_samples["mode"].append(mode)

    # Convert to arrays
    for key in all_samples:
        all_samples[key] = np.array(all_samples[key])

    n_samples = len(all_samples["target"])
    if n_samples < MIN_SAMPLES:
        return None, None

    # Temporal split
    split_idx = int(n_samples * TRAIN_RATIO)

    train = {k: v[:split_idx] for k, v in all_samples.items()}
    test = {k: v[split_idx:] for k, v in all_samples.items()}

    # Add time features
    for samples in [train, test]:
        hours = samples["hour"].astype(float)
        samples["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        samples["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    return train, test


# =============================================================================
# Baseline Models (same as original, but work on any train/test dict)
# =============================================================================

def B1_persistence(train: dict, test: dict) -> np.ndarray:
    """B1: Predict current temperature stays the same."""
    return test["indoor_temp"].copy()


def B2_thermal_drift(train: dict, test: dict) -> np.ndarray:
    """B2: Linear model on thermal driving force only."""
    if len(train["thermal_drive"]) < 2:
        return test["indoor_temp"].copy()

    X_train = train["thermal_drive"].reshape(-1, 1)
    X_test = test["thermal_drive"].reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, train["target"] - train["indoor_temp"])

    delta = model.predict(X_test)
    return test["indoor_temp"] + delta


def B3_mode_aware_target(train: dict, test: dict) -> np.ndarray:
    """B3: Drift toward mode-specific target with learned rates."""
    pred = np.zeros(len(test["target"]))

    for mode in ["heating", "cooling", "passive"]:
        train_mask = train["mode"] == mode
        test_mask = test["mode"] == mode

        if train_mask.sum() < 5 or test_mask.sum() == 0:
            pred[test_mask] = test["indoor_temp"][test_mask]
            continue

        if mode == "heating":
            train_target = train["heat_sp"][train_mask]
            test_target = test["heat_sp"][test_mask]
        elif mode == "cooling":
            train_target = train["cool_sp"][train_mask]
            test_target = test["cool_sp"][test_mask]
        else:
            train_target = train["outdoor_temp_future"][train_mask]
            test_target = test["outdoor_temp_future"][test_mask]

        train_indoor = train["indoor_temp"][train_mask]
        train_delta = train["target"][train_mask] - train_indoor
        train_gap = train_target - train_indoor

        k = np.dot(train_delta, train_gap) / (np.dot(train_gap, train_gap) + 1e-8)

        test_indoor = test["indoor_temp"][test_mask]
        test_gap = test_target - test_indoor
        pred[test_mask] = test_indoor + k * test_gap

    return pred


def B4_linreg_thermal(train: dict, test: dict) -> np.ndarray:
    """B4: Linear regression with thermal driving force."""
    if len(train["indoor_temp"]) < 5:
        return test["indoor_temp"].copy()

    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["heat_sp"],
            s["cool_sp"],
        ])

    X_train = features(train)
    X_test = features(test)

    # Use unscaled to avoid issues with zero variance
    model = LinearRegression()
    model.fit(X_train, train["target"])
    return model.predict(X_test)


def B5_linreg_mode(train: dict, test: dict) -> np.ndarray:
    """B5: Linear regression with mode indicators."""
    if len(train["indoor_temp"]) < 5:
        return test["indoor_temp"].copy()

    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["heat_sp"],
            s["cool_sp"],
            s["is_heating"],
            s["is_cooling"],
        ])

    X_train = features(train)
    X_test = features(test)

    model = LinearRegression()
    model.fit(X_train, train["target"])
    return model.predict(X_test)


def B6_linreg_gap(train: dict, test: dict) -> np.ndarray:
    """B6: Linear regression with gap to setpoint."""
    if len(train["indoor_temp"]) < 5:
        return test["indoor_temp"].copy()

    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["heat_sp"],
            s["cool_sp"],
            s["is_heating"],
            s["is_cooling"],
            s["gap_to_target"],
        ])

    X_train = features(train)
    X_test = features(test)

    model = LinearRegression()
    model.fit(X_train, train["target"])
    return model.predict(X_test)


def B7_linreg_time(train: dict, test: dict) -> np.ndarray:
    """B7: Linear regression with time features."""
    if len(train["indoor_temp"]) < 5:
        return test["indoor_temp"].copy()

    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["heat_sp"],
            s["cool_sp"],
            s["is_heating"],
            s["is_cooling"],
            s["gap_to_target"],
            s["hour_sin"],
            s["hour_cos"],
        ])

    X_train = features(train)
    X_test = features(test)

    model = LinearRegression()
    model.fit(X_train, train["target"])
    return model.predict(X_test)


def B9_per_mode_linreg(train: dict, test: dict) -> np.ndarray:
    """B9: Separate linear regression per mode."""
    pred = test["indoor_temp"].copy()  # Default to persistence

    def features(s, mode):
        base = [
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["hour_sin"],
            s["hour_cos"],
        ]
        if mode == "heating":
            base.append(s["heat_sp"])
            base.append(s["gap_to_target"])
        elif mode == "cooling":
            base.append(s["cool_sp"])
            base.append(s["gap_to_target"])
        return np.column_stack(base)

    for mode in ["heating", "cooling", "passive"]:
        train_mask = train["mode"] == mode
        test_mask = test["mode"] == mode

        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue

        train_mode = {k: v[train_mask] for k, v in train.items()}
        test_mode = {k: v[test_mask] for k, v in test.items()}

        X_train = features(train_mode, mode)
        X_test = features(test_mode, mode)

        model = LinearRegression()
        model.fit(X_train, train_mode["target"])
        pred[test_mask] = model.predict(X_test)

    return pred


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Per-Home Baseline Experiment (Oracle Upper Bound)")
    print("=" * 70)
    print(f"Horizon: {HORIZON} steps ({HORIZON * 5} min)")
    print(f"Sample rate: every {SAMPLE_RATE} steps ({SAMPLE_RATE * 5} min)")
    print(f"Train/test split: {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}% (temporal)")
    print(f"Min samples per home: {MIN_SAMPLES}")
    print()

    df = load_data()

    # Get all homes
    homes = df["home_id"].unique().to_list()
    print(f"Total homes: {len(homes)}")

    baselines = [
        ("B1: Persistence", B1_persistence),
        ("B2: Thermal Drift", B2_thermal_drift),
        ("B3: Mode-Aware Target", B3_mode_aware_target),
        ("B4: LinReg+Thermal", B4_linreg_thermal),
        ("B5: LinReg+Mode", B5_linreg_mode),
        ("B6: LinReg+Gap", B6_linreg_gap),
        ("B7: LinReg+Time", B7_linreg_time),
        ("B9: Per-Mode LinReg", B9_per_mode_linreg),
    ]

    # Aggregate predictions and targets
    all_preds = {name: [] for name, _ in baselines}
    all_targets = []
    all_modes = []
    all_states = []

    home_count = 0
    skipped = 0

    print()
    print("Processing homes...")
    for home_id in tqdm(homes, desc="Homes"):
        home_df = df.filter(pl.col("home_id") == home_id)
        state = home_df["state"][0]

        train, test = create_home_samples(home_df)

        if train is None or test is None:
            skipped += 1
            continue

        if len(test["target"]) < 10:
            skipped += 1
            continue

        home_count += 1

        # Run all baselines for this home
        for name, fn in baselines:
            try:
                pred = fn(train, test)
                all_preds[name].append(pred)
            except Exception as e:
                # Fallback to persistence if model fails
                all_preds[name].append(test["indoor_temp"].copy())

        all_targets.append(test["target"])
        all_modes.append(test["mode"])
        all_states.append(np.array([state] * len(test["target"])))

    print(f"\nProcessed {home_count} homes, skipped {skipped}")

    # Concatenate all results
    for name in all_preds:
        all_preds[name] = np.concatenate(all_preds[name])
    all_targets = np.concatenate(all_targets)
    all_modes = np.concatenate(all_modes)
    all_states = np.concatenate(all_states)

    print(f"Total test samples: {len(all_targets):,}")

    # Mode distribution
    print("\nMode distribution:")
    for mode in ["passive", "heating", "cooling"]:
        count = (all_modes == mode).sum()
        pct = 100 * count / len(all_modes)
        print(f"  {mode}: {count:,} ({pct:.1f}%)")

    # Compute metrics
    print()
    print("=" * 90)
    print("PER-HOME RESULTS (Oracle Upper Bound)")
    print("=" * 90)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'P95':>7} {'Bias':>7}")
    print("-" * 90)

    results = []
    for name, _ in baselines:
        pred = all_preds[name]
        errors = pred - all_targets
        abs_errors = np.abs(errors)

        mae = abs_errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        p95 = np.percentile(abs_errors, 95)
        bias = errors.mean()

        results.append((name, mae, rmse, p95, bias))
        print(f"{name:<22} {mae:>7.3f} {rmse:>7.3f} {p95:>7.3f} {bias:>+7.3f}")

    # Per-mode breakdown
    for mode in ["heating", "cooling", "passive"]:
        mask = all_modes == mode
        count = mask.sum()
        pct = 100 * count / len(all_modes)

        print()
        print("=" * 90)
        print(f"{mode.upper()} MODE ({count:,} samples, {pct:.1f}%)")
        print("=" * 90)
        print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'P95':>7} {'Bias':>7}")
        print("-" * 90)

        for name, _ in baselines:
            pred = all_preds[name][mask]
            target = all_targets[mask]
            errors = pred - target
            abs_errors = np.abs(errors)

            mae = abs_errors.mean()
            rmse = np.sqrt((errors ** 2).mean())
            p95 = np.percentile(abs_errors, 95)
            bias = errors.mean()

            print(f"{name:<22} {mae:>7.3f} {rmse:>7.3f} {p95:>7.3f} {bias:>+7.3f}")

    # Summary comparison with global results
    print()
    print("=" * 90)
    print("COMPARISON: Per-Home vs Global (from docs/BASELINE_RESULTS.md)")
    print("=" * 90)
    print(f"{'Model':<22} {'Per-Home MAE':>14} {'Global MAE':>14} {'Improvement':>14}")
    print("-" * 90)

    # Global MAEs from BASELINE_RESULTS.md
    global_mae = {
        "B1: Persistence": 0.434,
        "B2: Thermal Drift": 0.452,
        "B3: Mode-Aware Target": 0.489,
        "B4: LinReg+Thermal": 0.488,
        "B5: LinReg+Mode": 0.496,
        "B6: LinReg+Gap": 0.496,
        "B7: LinReg+Time": 0.502,
        "B9: Per-Mode LinReg": 0.504,
    }

    for name, mae, _, _, _ in results:
        g_mae = global_mae.get(name, 0)
        if g_mae > 0:
            improvement = 100 * (g_mae - mae) / g_mae
            print(f"{name:<22} {mae:>14.3f} {g_mae:>14.3f} {improvement:>+13.1f}%")

    print("=" * 90)


if __name__ == "__main__":
    main()
