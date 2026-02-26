"""
Baseline models for setpoint response prediction.

Predicts indoor temperature 15 minutes ahead during setpoint response episodes.

Usage: python scripts/run_setpoint_baselines.py
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from tqdm import tqdm
import time

# Config
DATA_PATH = "data/setpoint_responses.parquet"
HORIZON = 3  # 15 minutes ahead (3 timesteps at 5-min resolution)


@dataclass
class StratMetrics:
    """Metrics for a stratum."""
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    n_samples: int = 0


@dataclass
class Metrics:
    mae: float = 0.0
    rmse: float = 0.0
    p95: float = 0.0
    bias: float = 0.0
    by_type: Dict[str, StratMetrics] = field(default_factory=dict)
    by_gap: Dict[str, StratMetrics] = field(default_factory=dict)
    by_stage: Dict[str, StratMetrics] = field(default_factory=dict)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    change_types: np.ndarray,
    current_gaps: np.ndarray,
    timestep_idxs: np.ndarray,
) -> Metrics:
    """Compute all evaluation metrics."""
    errors = pred - target
    abs_errors = np.abs(errors)

    m = Metrics()
    m.mae = abs_errors.mean()
    m.rmse = np.sqrt((errors ** 2).mean())
    m.p95 = np.percentile(abs_errors, 95)
    m.bias = errors.mean()

    # By change type
    for ctype in ["heat_increase", "cool_decrease"]:
        mask = change_types == ctype
        if mask.sum() > 0:
            type_errors = pred[mask] - target[mask]
            m.by_type[ctype] = StratMetrics(
                mae=np.abs(type_errors).mean(),
                rmse=np.sqrt((type_errors ** 2).mean()),
                bias=type_errors.mean(),
                n_samples=mask.sum(),
            )

    # By current gap size
    gap_bins = [
        ("0-1°F", 0, 1),
        ("1-2°F", 1, 2),
        ("2-3°F", 2, 3),
        ("3-5°F", 3, 5),
        (">5°F", 5, 100),
    ]
    for name, lo, hi in gap_bins:
        mask = (np.abs(current_gaps) >= lo) & (np.abs(current_gaps) < hi)
        if mask.sum() > 0:
            gap_errors = pred[mask] - target[mask]
            m.by_gap[name] = StratMetrics(
                mae=np.abs(gap_errors).mean(),
                rmse=np.sqrt((gap_errors ** 2).mean()),
                bias=gap_errors.mean(),
                n_samples=mask.sum(),
            )

    # By episode stage
    stage_bins = [
        ("early (0-10min)", 0, 2),
        ("mid (10-30min)", 2, 6),
        ("late (>30min)", 6, 1000),
    ]
    for name, lo, hi in stage_bins:
        mask = (timestep_idxs >= lo) & (timestep_idxs < hi)
        if mask.sum() > 0:
            stage_errors = pred[mask] - target[mask]
            m.by_stage[name] = StratMetrics(
                mae=np.abs(stage_errors).mean(),
                rmse=np.sqrt((stage_errors ** 2).mean()),
                bias=stage_errors.mean(),
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


def create_within_home_splits(df: pl.DataFrame, train_frac: float = 0.7) -> tuple[dict, dict]:
    """Split episodes within each home for per-home evaluation.

    For each home, 70% of episodes go to train, 30% to test.
    This allows per-home models to learn from some episodes and test on others.
    """
    print(f"Creating within-home splits ({train_frac:.0%} train)...")
    t0 = time.time()

    np.random.seed(42)

    train_windows = {
        "current_temp": [], "target_setpoint": [], "start_temp": [],
        "initial_gap": [], "timestep_idx": [], "change_type": [],
        "outdoor_temp": [], "y_true": [], "current_gap": [],
        "elapsed_change": [], "home_id": [],
    }
    test_windows = {k: [] for k in train_windows}

    homes = df["home_id"].unique().to_list()

    for home in tqdm(homes, desc="  Splitting homes"):
        home_df = df.filter(pl.col("home_id") == home)
        episodes = home_df["episode_id"].unique().to_list()

        if len(episodes) < 2:
            continue

        # Shuffle and split episodes
        np.random.shuffle(episodes)
        n_train = max(1, int(len(episodes) * train_frac))
        train_episodes = set(episodes[:n_train])
        test_episodes = set(episodes[n_train:])

        # Process each episode
        for ep_id in episodes:
            ep = home_df.filter(pl.col("episode_id") == ep_id).sort("timestep_idx")

            temps = ep["Indoor_AverageTemperature"].to_numpy()
            n = len(temps)

            if n <= HORIZON:
                continue

            target_sp = ep["target_setpoint"][0]
            start_temp = ep["start_temp"][0]
            initial_gap = ep["initial_gap"][0]
            change_type = ep["change_type"][0]
            outdoor = ep["Outdoor_Temperature"].to_numpy()
            timestep_idxs = ep["timestep_idx"].to_numpy()

            # Choose target dict
            windows = train_windows if ep_id in train_episodes else test_windows

            # Create windows
            for t in range(n - HORIZON):
                curr_temp = temps[t]
                fut_temp = temps[t + HORIZON]

                if np.isnan(curr_temp) or np.isnan(fut_temp):
                    continue

                windows["current_temp"].append(curr_temp)
                windows["target_setpoint"].append(target_sp)
                windows["start_temp"].append(start_temp)
                windows["initial_gap"].append(initial_gap)
                windows["timestep_idx"].append(timestep_idxs[t])
                windows["change_type"].append(change_type)
                windows["outdoor_temp"].append(outdoor[t])
                windows["y_true"].append(fut_temp)
                windows["current_gap"].append(target_sp - curr_temp)
                windows["elapsed_change"].append(curr_temp - start_temp)
                windows["home_id"].append(home)

    # Convert to arrays
    for key in train_windows:
        train_windows[key] = np.array(train_windows[key])
        test_windows[key] = np.array(test_windows[key])

    print(f"  Train: {len(train_windows['y_true']):,} windows")
    print(f"  Test: {len(test_windows['y_true']):,} windows")
    print(f"  Homes in both: {len(set(train_windows['home_id']) & set(test_windows['home_id']))}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return train_windows, test_windows


def create_windows(df: pl.DataFrame, split: str) -> dict:
    """Create 15-minute prediction windows from episodes."""
    print(f"Creating {split} windows...")
    t0 = time.time()

    windows = {
        "current_temp": [],
        "target_setpoint": [],
        "start_temp": [],
        "initial_gap": [],
        "timestep_idx": [],
        "change_type": [],
        "outdoor_temp": [],
        "y_true": [],
        # Derived
        "current_gap": [],  # signed: target - current
        "elapsed_change": [],  # current - start
        "home_id": [],  # for per-home models
    }

    split_df = df.filter(pl.col("split") == split)
    episode_ids = split_df["episode_id"].unique().to_list()

    for ep_id in tqdm(episode_ids, desc=f"  {split} episodes"):
        ep = split_df.filter(pl.col("episode_id") == ep_id).sort("timestep_idx")

        temps = ep["Indoor_AverageTemperature"].to_numpy()
        n = len(temps)

        if n <= HORIZON:
            continue

        target_sp = ep["target_setpoint"][0]
        start_temp = ep["start_temp"][0]
        initial_gap = ep["initial_gap"][0]
        change_type = ep["change_type"][0]
        outdoor = ep["Outdoor_Temperature"].to_numpy()
        timestep_idxs = ep["timestep_idx"].to_numpy()
        home_id = ep["home_id"][0]

        # Create windows
        for t in range(n - HORIZON):
            curr_temp = temps[t]
            fut_temp = temps[t + HORIZON]

            if np.isnan(curr_temp) or np.isnan(fut_temp):
                continue

            windows["current_temp"].append(curr_temp)
            windows["target_setpoint"].append(target_sp)
            windows["start_temp"].append(start_temp)
            windows["initial_gap"].append(initial_gap)
            windows["timestep_idx"].append(timestep_idxs[t])
            windows["change_type"].append(change_type)
            windows["outdoor_temp"].append(outdoor[t])
            windows["y_true"].append(fut_temp)

            # Derived features
            current_gap = target_sp - curr_temp  # signed
            elapsed_change = curr_temp - start_temp
            windows["current_gap"].append(current_gap)
            windows["elapsed_change"].append(elapsed_change)
            windows["home_id"].append(home_id)

    # Convert to arrays
    for key in windows:
        windows[key] = np.array(windows[key])

    print(f"  {len(windows['y_true']):,} windows in {time.time()-t0:.1f}s")

    # Print distribution
    heat_mask = windows["change_type"] == "heat_increase"
    cool_mask = windows["change_type"] == "cool_decrease"
    print(f"  heat_increase: {heat_mask.sum():,} ({100*heat_mask.mean():.1f}%)")
    print(f"  cool_decrease: {cool_mask.sum():,} ({100*cool_mask.mean():.1f}%)")

    return windows


# =============================================================================
# Baseline Models
# =============================================================================

def B1_persistence(test: dict) -> np.ndarray:
    """B1: Predict temperature stays the same."""
    return test["current_temp"].copy()


def B2_target_reached(test: dict) -> np.ndarray:
    """B2: Predict temperature reaches setpoint."""
    return test["target_setpoint"].copy()


def B3_linear_interp(test: dict) -> np.ndarray:
    """B3: Extrapolate observed rate of change."""
    elapsed_min = test["timestep_idx"] * 5  # minutes into episode
    elapsed_change = test["elapsed_change"]

    # Compute rate (avoid division by zero)
    rate = np.zeros_like(elapsed_change)
    valid = elapsed_min > 0
    rate[valid] = elapsed_change[valid] / elapsed_min[valid]

    # Predict 15 min ahead
    return test["current_temp"] + rate * 15


def B4_gap_fraction(train: dict, test: dict) -> np.ndarray:
    """B4: Close a learned fraction of remaining gap."""
    # Learn k: minimize MAE for y = current + k * gap
    # Optimal k via least squares on (y_true - current) = k * gap
    delta_train = train["y_true"] - train["current_temp"]
    gap_train = train["current_gap"]

    # k = sum(delta * gap) / sum(gap^2)
    k = np.dot(delta_train, gap_train) / (np.dot(gap_train, gap_train) + 1e-8)
    print(f"    B4 learned k = {k:.4f}")

    return test["current_temp"] + k * test["current_gap"]


def B5_clipped_gap(train: dict, test: dict) -> np.ndarray:
    """B5: Gap fraction but clipped to not overshoot target."""
    # Learn k same as B4
    delta_train = train["y_true"] - train["current_temp"]
    gap_train = train["current_gap"]
    k = np.dot(delta_train, gap_train) / (np.dot(gap_train, gap_train) + 1e-8)
    print(f"    B5 learned k = {k:.4f}")

    pred = test["current_temp"] + k * test["current_gap"]

    # Clip to not overshoot
    heat_mask = test["change_type"] == "heat_increase"
    cool_mask = test["change_type"] == "cool_decrease"

    # For heating: pred <= target
    pred[heat_mask] = np.minimum(pred[heat_mask], test["target_setpoint"][heat_mask])
    # For cooling: pred >= target
    pred[cool_mask] = np.maximum(pred[cool_mask], test["target_setpoint"][cool_mask])

    return pred


def B6_mode_gap(train: dict, test: dict) -> np.ndarray:
    """B6: Separate gap fraction for heating vs cooling."""
    pred = np.zeros(len(test["y_true"]))

    for ctype in ["heat_increase", "cool_decrease"]:
        train_mask = train["change_type"] == ctype
        test_mask = test["change_type"] == ctype

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            pred[test_mask] = test["current_temp"][test_mask]
            continue

        delta = train["y_true"][train_mask] - train["current_temp"][train_mask]
        gap = train["current_gap"][train_mask]
        k = np.dot(delta, gap) / (np.dot(gap, gap) + 1e-8)
        print(f"    B6 {ctype}: k = {k:.4f}")

        pred[test_mask] = (
            test["current_temp"][test_mask] +
            k * test["current_gap"][test_mask]
        )

    return pred


def B7_time_aware_gap(train: dict, test: dict) -> np.ndarray:
    """B7: Gap fraction varies by episode stage."""
    pred = np.zeros(len(test["y_true"]))

    stages = [
        ("early", 0, 2),   # 0-10 min
        ("mid", 2, 6),     # 10-30 min
        ("late", 6, 1000), # >30 min
    ]

    for stage_name, lo, hi in stages:
        train_mask = (train["timestep_idx"] >= lo) & (train["timestep_idx"] < hi)
        test_mask = (test["timestep_idx"] >= lo) & (test["timestep_idx"] < hi)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            pred[test_mask] = test["current_temp"][test_mask]
            continue

        delta = train["y_true"][train_mask] - train["current_temp"][train_mask]
        gap = train["current_gap"][train_mask]
        k = np.dot(delta, gap) / (np.dot(gap, gap) + 1e-8)
        # Clamp k to reasonable range to avoid extrapolation blowup
        k = np.clip(k, 0, 1)
        print(f"    B7 {stage_name}: k = {k:.4f}")

        pred[test_mask] = (
            test["current_temp"][test_mask] +
            k * test["current_gap"][test_mask]
        )

    return pred


def B8_thermal_drive(train: dict, test: dict) -> np.ndarray:
    """B8: Predict based on thermal driving force (outdoor - indoor)."""
    from sklearn.linear_model import LinearRegression

    thermal_drive = train["outdoor_temp"] - train["current_temp"]
    delta = train["y_true"] - train["current_temp"]

    model = LinearRegression()
    model.fit(thermal_drive.reshape(-1, 1), delta)

    test_drive = test["outdoor_temp"] - test["current_temp"]
    pred_delta = model.predict(test_drive.reshape(-1, 1))

    print(f"    B8 coef = {model.coef_[0]:.4f}, intercept = {model.intercept_:.4f}")
    return test["current_temp"] + pred_delta


def B9_thermal_drive_plus_gap(train: dict, test: dict) -> np.ndarray:
    """B9: Combine thermal drive and gap to target."""
    from sklearn.linear_model import LinearRegression

    X_train = np.column_stack([
        train["outdoor_temp"] - train["current_temp"],  # thermal drive
        train["current_gap"],  # gap to target
    ])
    delta = train["y_true"] - train["current_temp"]

    model = LinearRegression()
    model.fit(X_train, delta)

    X_test = np.column_stack([
        test["outdoor_temp"] - test["current_temp"],
        test["current_gap"],
    ])
    pred_delta = model.predict(X_test)

    print(f"    B9 coefs: thermal={model.coef_[0]:.4f}, gap={model.coef_[1]:.4f}")
    return test["current_temp"] + pred_delta


def B10_per_home_gap(train: dict, test: dict, home_ids_train: np.ndarray, home_ids_test: np.ndarray) -> np.ndarray:
    """B10: Per-home gap fraction model."""
    pred = np.zeros(len(test["y_true"]))

    # Get unique homes in test set
    test_homes = np.unique(home_ids_test)

    # Learn k for each home, with fallback to global k
    delta_all = train["y_true"] - train["current_temp"]
    gap_all = train["current_gap"]
    k_global = np.dot(delta_all, gap_all) / (np.dot(gap_all, gap_all) + 1e-8)

    home_ks = {}
    for home in test_homes:
        train_mask = home_ids_train == home
        if train_mask.sum() >= 10:  # Need enough samples
            delta = train["y_true"][train_mask] - train["current_temp"][train_mask]
            gap = train["current_gap"][train_mask]
            k = np.dot(delta, gap) / (np.dot(gap, gap) + 1e-8)
            # Clamp to reasonable range
            k = np.clip(k, 0, 0.5)
            home_ks[home] = k
        else:
            home_ks[home] = k_global

    # Apply per-home k
    for home in test_homes:
        test_mask = home_ids_test == home
        k = home_ks[home]
        pred[test_mask] = (
            test["current_temp"][test_mask] +
            k * test["current_gap"][test_mask]
        )

    print(f"    B10: {len(home_ks)} homes, k range [{min(home_ks.values()):.3f}, {max(home_ks.values()):.3f}]")
    return pred


def B11_per_home_thermal_gap(train: dict, test: dict, home_ids_train: np.ndarray, home_ids_test: np.ndarray) -> np.ndarray:
    """B11: Per-home model with thermal drive + gap."""
    from sklearn.linear_model import Ridge

    pred = np.zeros(len(test["y_true"]))

    # Global fallback model
    X_all = np.column_stack([
        train["outdoor_temp"] - train["current_temp"],
        train["current_gap"],
    ])
    delta_all = train["y_true"] - train["current_temp"]
    global_model = Ridge(alpha=1.0)
    global_model.fit(X_all, delta_all)

    test_homes = np.unique(home_ids_test)
    n_home_models = 0

    for home in test_homes:
        train_mask = home_ids_train == home
        test_mask = home_ids_test == home

        if train_mask.sum() >= 20:  # Need enough samples for 2-param model
            X_train = np.column_stack([
                train["outdoor_temp"][train_mask] - train["current_temp"][train_mask],
                train["current_gap"][train_mask],
            ])
            delta = train["y_true"][train_mask] - train["current_temp"][train_mask]

            model = Ridge(alpha=1.0)
            model.fit(X_train, delta)
            n_home_models += 1
        else:
            model = global_model

        X_test = np.column_stack([
            test["outdoor_temp"][test_mask] - test["current_temp"][test_mask],
            test["current_gap"][test_mask],
        ])
        pred[test_mask] = test["current_temp"][test_mask] + model.predict(X_test)

    print(f"    B11: {n_home_models}/{len(test_homes)} homes with own model")
    return pred


# =============================================================================
# Main
# =============================================================================

def run_evaluation(train: dict, test: dict, title: str):
    """Run all baselines and print results."""
    print()
    print("Running baselines...")
    print("-" * 70)

    results = []

    baselines = [
        ("B1: Persistence", lambda: B1_persistence(test)),
        ("B2: Target Reached", lambda: B2_target_reached(test)),
        ("B3: Linear Interp", lambda: B3_linear_interp(test)),
        ("B4: Gap Fraction", lambda: B4_gap_fraction(train, test)),
        ("B5: Clipped Gap", lambda: B5_clipped_gap(train, test)),
        ("B6: Mode-Specific Gap", lambda: B6_mode_gap(train, test)),
        ("B7: Time-Aware Gap", lambda: B7_time_aware_gap(train, test)),
        ("B8: Thermal Drive", lambda: B8_thermal_drive(train, test)),
        ("B9: Thermal+Gap", lambda: B9_thermal_drive_plus_gap(train, test)),
        ("B10: Per-Home Gap", lambda: B10_per_home_gap(train, test, train["home_id"], test["home_id"])),
        ("B11: Per-Home Thermal", lambda: B11_per_home_thermal_gap(train, test, train["home_id"], test["home_id"])),
    ]

    for name, fn in baselines:
        print(f"  {name}...")
        t0 = time.time()
        pred = fn()
        metrics = compute_metrics(
            pred, test["y_true"],
            test["change_type"], test["current_gap"], test["timestep_idx"]
        )
        elapsed = time.time() - t0
        results.append((name, metrics, elapsed))

    # Print overall results
    print()
    print("=" * 80)
    print("Overall Results")
    print("=" * 80)
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'P95':>8} {'Bias':>8}")
    print("-" * 80)
    for name, m, _ in results:
        print(f"{name:<22} {m.mae:>8.3f} {m.rmse:>8.3f} {m.p95:>8.3f} {m.bias:>+8.3f}")

    # By change type
    print()
    print("=" * 80)
    print("Results by Change Type")
    print("=" * 80)
    print(f"{'Model':<22} {'heat_increase':>14} {'cool_decrease':>14}")
    print("-" * 80)
    for name, m, _ in results:
        h = m.by_type.get("heat_increase")
        c = m.by_type.get("cool_decrease")
        h_mae = f"{h.mae:.3f}" if h else "N/A"
        c_mae = f"{c.mae:.3f}" if c else "N/A"
        print(f"{name:<22} {h_mae:>14} {c_mae:>14}")

    # Sample counts by type
    h = results[0][1].by_type.get("heat_increase")
    c = results[0][1].by_type.get("cool_decrease")
    print(f"{'(n samples)':<22} {h.n_samples if h else 0:>14,} {c.n_samples if c else 0:>14,}")

    # By gap size
    print()
    print("=" * 80)
    print("Results by Current Gap Size")
    print("=" * 80)
    gap_names = ["0-1°F", "1-2°F", "2-3°F", "3-5°F", ">5°F"]
    header = f"{'Model':<22}" + "".join(f"{g:>10}" for g in gap_names)
    print(header)
    print("-" * 80)
    for name, m, _ in results:
        row = f"{name:<22}"
        for g in gap_names:
            gm = m.by_gap.get(g)
            row += f"{gm.mae:>10.3f}" if gm else f"{'N/A':>10}"
        print(row)

    # Sample counts by gap
    row = f"{'(n samples)':<22}"
    for g in gap_names:
        gm = results[0][1].by_gap.get(g)
        row += f"{gm.n_samples if gm else 0:>10,}"
    print(row)

    # By episode stage
    print()
    print("=" * 80)
    print("Results by Episode Stage")
    print("=" * 80)
    stage_names = ["early (0-10min)", "mid (10-30min)", "late (>30min)"]
    header = f"{'Model':<22}" + "".join(f"{s:>18}" for s in stage_names)
    print(header)
    print("-" * 80)
    for name, m, _ in results:
        row = f"{name:<22}"
        for s in stage_names:
            sm = m.by_stage.get(s)
            row += f"{sm.mae:>18.3f}" if sm else f"{'N/A':>18}"
        print(row)

    # Sample counts by stage
    row = f"{'(n samples)':<22}"
    for s in stage_names:
        sm = results[0][1].by_stage.get(s)
        row += f"{sm.n_samples if sm else 0:>18,}"
    print(row)

    print()
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setpoint response baselines")
    parser.add_argument("--within-home", action="store_true",
                        help="Split episodes within each home (for per-home model eval)")
    args = parser.parse_args()

    print("=" * 70)
    print("Setpoint Response Baselines")
    print("=" * 70)
    print(f"Horizon: {HORIZON} steps ({HORIZON * 5} min)")

    df = load_data()

    if args.within_home:
        print()
        print("MODE: Within-home episode split")
        print("  (train/test on different episodes from SAME homes)")
        train, test = create_within_home_splits(df, train_frac=0.7)
        run_evaluation(train, test, "Within-Home Split")
    else:
        print()
        print("MODE: Cross-home split (original dataset splits)")
        print("  (train/test on DIFFERENT homes)")
        train = create_windows(df, "train")
        test = create_windows(df, "test")
        run_evaluation(train, test, "Cross-Home Split")


if __name__ == "__main__":
    main()
