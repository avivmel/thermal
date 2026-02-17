"""
Baseline models for thermal prediction.

Usage: python scripts/run_baselines.py
"""

import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, Tuple
from tqdm import tqdm
import time

# Config
DATA_PATH = "data/thermal_dataset.csv"
HORIZON = 6  # 30 minutes ahead
CONTEXT = 6  # 30 minutes of history
SAMPLE_RATE = 72  # Sample every 72 timesteps (6 hours)


@dataclass
class ModeMetrics:
    """Metrics for a single mode."""
    mae: float = 0.0
    rmse: float = 0.0
    p95: float = 0.0
    bias: float = 0.0
    n_samples: int = 0


@dataclass
class Metrics:
    mae: float = 0.0
    rmse: float = 0.0
    max_error: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    bias: float = 0.0
    r2: float = 0.0
    by_state: Dict[str, float] = field(default_factory=dict)
    by_mode: Dict[str, ModeMetrics] = field(default_factory=dict)


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    states: np.ndarray, modes: np.ndarray) -> Metrics:
    """Compute all evaluation metrics."""
    errors = pred - target
    abs_errors = np.abs(errors)

    m = Metrics()
    m.mae = abs_errors.mean()
    m.rmse = np.sqrt((errors ** 2).mean())
    m.max_error = abs_errors.max()
    m.p90 = np.percentile(abs_errors, 90)
    m.p95 = np.percentile(abs_errors, 95)
    m.bias = errors.mean()

    ss_res = (errors ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    m.r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Per-state MAE
    for state in np.unique(states):
        mask = states == state
        if mask.sum() > 0:
            m.by_state[state] = np.abs(pred[mask] - target[mask]).mean()

    # Per-mode full metrics
    for mode in np.unique(modes):
        mask = modes == mode
        if mask.sum() > 0:
            mode_errors = pred[mask] - target[mask]
            mode_abs = np.abs(mode_errors)
            m.by_mode[mode] = ModeMetrics(
                mae=mode_abs.mean(),
                rmse=np.sqrt((mode_errors ** 2).mean()),
                p95=np.percentile(mode_abs, 95),
                bias=mode_errors.mean(),
                n_samples=mask.sum(),
            )

    return m


def load_data() -> pl.DataFrame:
    """Load data with Polars."""
    print("Loading data...")
    t0 = time.time()
    df = pl.scan_csv(DATA_PATH).collect()
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def create_samples(df: pl.DataFrame, split: str) -> dict:
    """Create windowed samples for a given split."""
    print(f"Creating {split} samples...")
    t0 = time.time()

    samples = {
        "indoor_temp": [],
        "outdoor_temp": [],
        "outdoor_temp_future": [],
        "heat_sp": [],
        "cool_sp": [],
        "target": [],
        "state": [],
        "hour": [],
        # Derived features
        "thermal_drive": [],      # outdoor - indoor
        "is_heating": [],
        "is_cooling": [],
        "gap_to_target": [],
        "mode": [],
    }

    split_df = df.filter(pl.col("split") == split)
    homes = split_df["home_id"].unique().to_list()

    for home_id in tqdm(homes, desc=f"  {split} homes"):
        home_df = df.filter(pl.col("home_id") == home_id).sort("timestamp")

        indoor = home_df["Indoor_AverageTemperature"].to_numpy()
        outdoor = home_df["Outdoor_Temperature"].to_numpy()
        heat = home_df["Indoor_HeatSetpoint"].to_numpy()
        cool = home_df["Indoor_CoolSetpoint"].to_numpy()
        state = home_df["state"][0]

        # Extract hour from timestamp
        timestamps = home_df["timestamp"].to_list()

        n = len(indoor)
        valid_start = CONTEXT - 1
        valid_end = n - HORIZON

        if valid_end <= valid_start:
            continue

        indices = np.arange(valid_start, valid_end, SAMPLE_RATE)

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
                hour = 12  # Default

            samples["indoor_temp"].append(curr_indoor)
            samples["outdoor_temp"].append(curr_outdoor)
            samples["outdoor_temp_future"].append(fut_outdoor)
            samples["heat_sp"].append(curr_heat)
            samples["cool_sp"].append(curr_cool)
            samples["target"].append(fut_indoor)
            samples["state"].append(state)
            samples["hour"].append(hour)
            samples["thermal_drive"].append(curr_outdoor - curr_indoor)
            samples["is_heating"].append(is_heating)
            samples["is_cooling"].append(is_cooling)
            samples["gap_to_target"].append(gap)
            samples["mode"].append(mode)

    for key in samples:
        samples[key] = np.array(samples[key])

    # Add time features
    hours = samples["hour"].astype(float)
    samples["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    samples["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    print(f"  {len(samples['target']):,} samples in {time.time()-t0:.1f}s")

    # Print mode distribution
    modes, counts = np.unique(samples["mode"], return_counts=True)
    print(f"  Mode distribution: ", end="")
    for m, c in zip(modes, counts):
        print(f"{m}={100*c/len(samples['mode']):.1f}% ", end="")
    print()

    return samples


# =============================================================================
# Baseline Models
# =============================================================================

def B1_persistence(test: dict) -> np.ndarray:
    """B1: Predict current temperature stays the same."""
    return test["indoor_temp"].copy()


def B2_thermal_drift(train: dict, test: dict) -> np.ndarray:
    """B2: Linear model on thermal driving force only."""
    X_train = train["thermal_drive"].reshape(-1, 1)
    X_test = test["thermal_drive"].reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, train["target"] - train["indoor_temp"])

    delta = model.predict(X_test)
    return test["indoor_temp"] + delta


def B3_mode_aware_target(train: dict, test: dict) -> np.ndarray:
    """B3: Drift toward mode-specific target with learned rates."""
    pred = np.zeros(len(test["target"]))

    # Learn drift rates per mode
    for mode in ["heating", "cooling", "passive"]:
        train_mask = train["mode"] == mode
        test_mask = test["mode"] == mode

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            pred[test_mask] = test["indoor_temp"][test_mask]
            continue

        # Determine target
        if mode == "heating":
            train_target = train["heat_sp"][train_mask]
            test_target = test["heat_sp"][test_mask]
        elif mode == "cooling":
            train_target = train["cool_sp"][train_mask]
            test_target = test["cool_sp"][test_mask]
        else:  # passive
            train_target = train["outdoor_temp_future"][train_mask]
            test_target = test["outdoor_temp_future"][test_mask]

        # Learn drift rate: delta = k * (target - indoor)
        train_indoor = train["indoor_temp"][train_mask]
        train_delta = train["target"][train_mask] - train_indoor
        train_gap = train_target - train_indoor

        # Fit k via least squares
        k = np.dot(train_delta, train_gap) / (np.dot(train_gap, train_gap) + 1e-8)

        # Predict
        test_indoor = test["indoor_temp"][test_mask]
        test_gap = test_target - test_indoor
        pred[test_mask] = test_indoor + k * test_gap

    return pred


def B4_linreg_thermal(train: dict, test: dict) -> np.ndarray:
    """B4: Linear regression with thermal driving force."""
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, train["target"])
    return model.predict(X_test_scaled)


def B5_linreg_mode(train: dict, test: dict) -> np.ndarray:
    """B5: Linear regression with mode indicators."""
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, train["target"])
    return model.predict(X_test_scaled)


def B6_linreg_gap(train: dict, test: dict) -> np.ndarray:
    """B6: Linear regression with gap to setpoint."""
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, train["target"])
    return model.predict(X_test_scaled)


def B7_linreg_time(train: dict, test: dict) -> np.ndarray:
    """B7: Linear regression with time features."""
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, train["target"])
    return model.predict(X_test_scaled)


def B8_linreg_full(train: dict, test: dict) -> np.ndarray:
    """B8: Full linear regression (same as B7, all features)."""
    return B7_linreg_time(train, test)


def B9_per_mode_linreg(train: dict, test: dict) -> np.ndarray:
    """B9: Separate linear regression per mode."""
    pred = np.zeros(len(test["target"]))

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

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            pred[test_mask] = test["indoor_temp"][test_mask]
            continue

        # Extract mode-specific data
        train_mode = {k: v[train_mask] for k, v in train.items()}
        test_mode = {k: v[test_mask] for k, v in test.items()}

        X_train = features(train_mode, mode)
        X_test = features(test_mode, mode)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, train_mode["target"])
        pred[test_mask] = model.predict(X_test_scaled)

    return pred


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Thermal Prediction Baselines")
    print("=" * 70)
    print(f"Horizon: {HORIZON} steps ({HORIZON * 5} min)")
    print(f"Sample rate: every {SAMPLE_RATE} steps ({SAMPLE_RATE * 5 / 60:.1f} hrs)")
    print()

    df = load_data()
    train = create_samples(df, "train")
    test = create_samples(df, "test")

    print()
    print("Running baselines...")
    print("-" * 70)

    results = []

    baselines = [
        ("B1: Persistence", lambda: B1_persistence(test)),
        ("B2: Thermal Drift", lambda: B2_thermal_drift(train, test)),
        ("B3: Mode-Aware Target", lambda: B3_mode_aware_target(train, test)),
        ("B4: LinReg+Thermal", lambda: B4_linreg_thermal(train, test)),
        ("B5: LinReg+Mode", lambda: B5_linreg_mode(train, test)),
        ("B6: LinReg+Gap", lambda: B6_linreg_gap(train, test)),
        ("B7: LinReg+Time", lambda: B7_linreg_time(train, test)),
        ("B8: LinReg Full", lambda: B8_linreg_full(train, test)),
        ("B9: Per-Mode LinReg", lambda: B9_per_mode_linreg(train, test)),
    ]

    for name, fn in tqdm(baselines, desc="Baselines"):
        t0 = time.time()
        pred = fn()
        metrics = compute_metrics(pred, test["target"], test["state"], test["mode"])
        elapsed = time.time() - t0
        results.append((name, metrics, elapsed))

    # Get mode sample counts from first result
    mode_counts = {mode: results[0][1].by_mode[mode].n_samples
                   for mode in ["heating", "cooling", "passive"]}
    total = sum(mode_counts.values())

    # Print overall results
    print()
    print("=" * 90)
    print("Overall Results (all modes combined)")
    print("=" * 90)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'Max':>7} {'P95':>7} {'Bias':>7} {'R²':>7}")
    print("-" * 90)
    for name, m, _ in results:
        print(f"{name:<22} {m.mae:>7.3f} {m.rmse:>7.3f} {m.max_error:>7.2f} {m.p95:>7.3f} {m.bias:>+7.3f} {m.r2:>7.3f}")

    # Per-mode results - HEATING
    print()
    print("=" * 90)
    n, pct = mode_counts["heating"], 100 * mode_counts["heating"] / total
    print(f"HEATING MODE - HVAC warming toward setpoint ({n:,} samples, {pct:.1f}%)")
    print("=" * 90)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'P95':>7} {'Bias':>7}")
    print("-" * 90)
    for name, m, _ in results:
        mm = m.by_mode.get("heating")
        if mm:
            print(f"{name:<22} {mm.mae:>7.3f} {mm.rmse:>7.3f} {mm.p95:>7.3f} {mm.bias:>+7.3f}")

    # Per-mode results - COOLING
    print()
    print("=" * 90)
    n, pct = mode_counts["cooling"], 100 * mode_counts["cooling"] / total
    print(f"COOLING MODE - HVAC cooling toward setpoint ({n:,} samples, {pct:.1f}%)")
    print("=" * 90)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'P95':>7} {'Bias':>7}")
    print("-" * 90)
    for name, m, _ in results:
        mm = m.by_mode.get("cooling")
        if mm:
            print(f"{name:<22} {mm.mae:>7.3f} {mm.rmse:>7.3f} {mm.p95:>7.3f} {mm.bias:>+7.3f}")

    # Per-mode results - PASSIVE
    print()
    print("=" * 90)
    n, pct = mode_counts["passive"], 100 * mode_counts["passive"] / total
    print(f"PASSIVE MODE - HVAC off, temp drifting ({n:,} samples, {pct:.1f}%)")
    print("=" * 90)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'P95':>7} {'Bias':>7}")
    print("-" * 90)
    for name, m, _ in results:
        mm = m.by_mode.get("passive")
        if mm:
            print(f"{name:<22} {mm.mae:>7.3f} {mm.rmse:>7.3f} {mm.p95:>7.3f} {mm.bias:>+7.3f}")

    # Summary comparison
    print()
    print("=" * 90)
    print("SUMMARY: MAE by Mode")
    print("=" * 90)
    print(f"{'Model':<22} {'Heating':>10} {'Cooling':>10} {'Passive':>10} {'Overall':>10}")
    print("-" * 90)
    for name, m, _ in results:
        h = m.by_mode.get("heating")
        c = m.by_mode.get("cooling")
        p = m.by_mode.get("passive")
        print(f"{name:<22} {h.mae if h else 0:>10.3f} {c.mae if c else 0:>10.3f} {p.mae if p else 0:>10.3f} {m.mae:>10.3f}")

    print("=" * 90)


if __name__ == "__main__":
    main()
