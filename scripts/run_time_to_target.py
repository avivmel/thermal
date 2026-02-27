"""
Time-to-target prediction baselines.

Predicts minutes remaining until temperature reaches setpoint.

Usage: python scripts/run_time_to_target.py
"""

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from dataclasses import dataclass, field
from typing import Dict
from tqdm import tqdm
import time

# Config
DATA_PATH = "data/setpoint_responses.parquet"


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
    p95: float = 0.0
    bias: float = 0.0
    by_type: Dict[str, StratMetrics] = field(default_factory=dict)
    by_gap: Dict[str, StratMetrics] = field(default_factory=dict)
    by_stage: Dict[str, StratMetrics] = field(default_factory=dict)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    change_types: np.ndarray,
    initial_gaps: np.ndarray,
    elapsed_times: np.ndarray,
) -> Metrics:
    """Compute all evaluation metrics."""
    errors = pred - target
    abs_errors = np.abs(errors)

    # Avoid division by zero for MAPE
    valid_target = target > 1  # At least 1 minute remaining

    m = Metrics()
    m.mae = abs_errors.mean()
    m.rmse = np.sqrt((errors ** 2).mean())
    m.mape = 100 * (abs_errors[valid_target] / target[valid_target]).mean() if valid_target.sum() > 0 else 0
    m.p95 = np.percentile(abs_errors, 95)
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

    # By elapsed time (episode stage)
    stage_bins = [
        ("start (0-5min)", 0, 5),
        ("early (5-15min)", 5, 15),
        ("mid (15-30min)", 15, 30),
        ("late (>30min)", 30, 10000),
    ]
    for name, lo, hi in stage_bins:
        mask = (elapsed_times >= lo) & (elapsed_times < hi)
        if mask.sum() > 0:
            stage_errors = pred[mask] - target[mask]
            valid = target[mask] > 1
            m.by_stage[name] = StratMetrics(
                mae=np.abs(stage_errors).mean(),
                rmse=np.sqrt((stage_errors ** 2).mean()),
                mape=100 * (np.abs(stage_errors[valid]) / target[mask][valid]).mean() if valid.sum() > 0 else 0,
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


def create_samples(df: pl.DataFrame, train_frac: float = 0.7) -> tuple[dict, dict]:
    """Create time-to-target samples with within-home episode split."""
    print(f"Creating samples (within-home split, {train_frac:.0%} train)...")
    t0 = time.time()

    np.random.seed(42)

    train = {
        "current_temp": [], "target_setpoint": [], "current_gap": [],
        "outdoor_temp": [], "thermal_drive": [], "elapsed_time": [],
        "time_remaining": [], "change_type": [], "home_id": [],
        "initial_gap": [], "episode_duration": [],
    }
    test = {k: [] for k in train}

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

            temps = ep["Indoor_AverageTemperature"].to_numpy()
            outdoor = ep["Outdoor_Temperature"].to_numpy()
            target_sp = ep["target_setpoint"][0]
            initial_gap = ep["initial_gap"][0]
            change_type = ep["change_type"][0]

            n_steps = len(temps)
            episode_duration = n_steps * 5  # minutes

            # Choose target dict
            samples = train if ep_id in train_episodes else test

            # Create samples at each timestep
            for t in range(n_steps):
                curr_temp = temps[t]
                curr_outdoor = outdoor[t]

                if np.isnan(curr_temp) or np.isnan(curr_outdoor):
                    continue

                elapsed_time = t * 5  # minutes
                time_remaining = episode_duration - elapsed_time

                # Skip if already at end
                if time_remaining <= 0:
                    continue

                current_gap = target_sp - curr_temp
                thermal_drive = curr_outdoor - curr_temp

                samples["current_temp"].append(curr_temp)
                samples["target_setpoint"].append(target_sp)
                samples["current_gap"].append(current_gap)
                samples["outdoor_temp"].append(curr_outdoor)
                samples["thermal_drive"].append(thermal_drive)
                samples["elapsed_time"].append(elapsed_time)
                samples["time_remaining"].append(time_remaining)
                samples["change_type"].append(change_type)
                samples["home_id"].append(home)
                samples["initial_gap"].append(initial_gap)
                samples["episode_duration"].append(episode_duration)

    # Convert to arrays
    for key in train:
        train[key] = np.array(train[key])
        test[key] = np.array(test[key])

    print(f"  Train: {len(train['time_remaining']):,} samples")
    print(f"  Test: {len(test['time_remaining']):,} samples")
    print(f"  Homes in both: {len(set(train['home_id']) & set(test['home_id']))}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Stats
    print(f"\n  Target stats (time_remaining):")
    print(f"    Train - Mean: {train['time_remaining'].mean():.1f} min, Median: {np.median(train['time_remaining']):.1f} min")
    print(f"    Test  - Mean: {test['time_remaining'].mean():.1f} min, Median: {np.median(test['time_remaining']):.1f} min")

    return train, test


# =============================================================================
# Baseline Models
# =============================================================================

def B1_global_mean(train: dict, test: dict) -> np.ndarray:
    """B1: Predict global mean time remaining."""
    mean_time = train["time_remaining"].mean()
    print(f"    B1: mean = {mean_time:.1f} min")
    return np.full(len(test["time_remaining"]), mean_time)


def B2_gap_proportional(train: dict, test: dict) -> np.ndarray:
    """B2: Time proportional to current gap."""
    # Learn k: time = k * |gap|
    gaps = np.abs(train["current_gap"])
    times = train["time_remaining"]

    # k = sum(time * gap) / sum(gap^2)
    k = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)
    print(f"    B2: k = {k:.1f} min/°F")

    return k * np.abs(test["current_gap"])


def B3_gap_thermal(train: dict, test: dict) -> np.ndarray:
    """B3: Gap + thermal drive."""
    X_train = np.column_stack([
        np.abs(train["current_gap"]),
        train["thermal_drive"],
    ])
    X_test = np.column_stack([
        np.abs(test["current_gap"]),
        test["thermal_drive"],
    ])

    model = Ridge(alpha=1.0)
    model.fit(X_train, train["time_remaining"])

    print(f"    B3: coefs = gap:{model.coef_[0]:.1f}, thermal:{model.coef_[1]:.2f}")
    return np.maximum(0, model.predict(X_test))


def B4_mode_specific(train: dict, test: dict) -> np.ndarray:
    """B4: Separate k for heating vs cooling."""
    pred = np.zeros(len(test["time_remaining"]))

    for ctype in ["heat_increase", "cool_decrease"]:
        train_mask = train["change_type"] == ctype
        test_mask = test["change_type"] == ctype

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        gaps = np.abs(train["current_gap"][train_mask])
        times = train["time_remaining"][train_mask]
        k = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)

        print(f"    B4 {ctype}: k = {k:.1f} min/°F")
        pred[test_mask] = k * np.abs(test["current_gap"][test_mask])

    return pred


def B5_per_home_gap(train: dict, test: dict) -> np.ndarray:
    """B5: Per-home gap-proportional model."""
    pred = np.zeros(len(test["time_remaining"]))

    # Global fallback
    gaps_all = np.abs(train["current_gap"])
    times_all = train["time_remaining"]
    k_global = np.dot(times_all, gaps_all) / (np.dot(gaps_all, gaps_all) + 1e-8)

    test_homes = np.unique(test["home_id"])
    home_ks = {}

    for home in test_homes:
        train_mask = train["home_id"] == home

        if train_mask.sum() >= 10:
            gaps = np.abs(train["current_gap"][train_mask])
            times = train["time_remaining"][train_mask]
            k = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)
            # Clamp to reasonable range
            k = np.clip(k, 5, 60)  # 5-60 min per °F
            home_ks[home] = k
        else:
            home_ks[home] = k_global

    for home in test_homes:
        test_mask = test["home_id"] == home
        pred[test_mask] = home_ks[home] * np.abs(test["current_gap"][test_mask])

    k_vals = list(home_ks.values())
    print(f"    B5: {len(home_ks)} homes, k range [{min(k_vals):.1f}, {max(k_vals):.1f}] min/°F")
    return pred


def B6_per_home_linear(train: dict, test: dict) -> np.ndarray:
    """B6: Per-home linear model with multiple features."""
    pred = np.zeros(len(test["time_remaining"]))

    def features(d):
        return np.column_stack([
            np.abs(d["current_gap"]),
            d["thermal_drive"],
            d["elapsed_time"],
        ])

    # Global fallback
    X_all = features(train)
    global_model = Ridge(alpha=1.0)
    global_model.fit(X_all, train["time_remaining"])

    test_homes = np.unique(test["home_id"])
    n_home_models = 0

    for home in test_homes:
        train_mask = train["home_id"] == home
        test_mask = test["home_id"] == home

        if train_mask.sum() >= 30:
            train_home = {k: v[train_mask] for k, v in train.items()}
            X_train = features(train_home)

            model = Ridge(alpha=1.0)
            model.fit(X_train, train_home["time_remaining"])
            n_home_models += 1
        else:
            model = global_model

        test_home = {k: v[test_mask] for k, v in test.items()}
        X_test = features(test_home)
        pred[test_mask] = np.maximum(0, model.predict(X_test))

    print(f"    B6: {n_home_models}/{len(test_homes)} homes with own model")
    return pred


def B7_per_home_mode(train: dict, test: dict) -> np.ndarray:
    """B7: Per-home AND per-mode model."""
    pred = np.zeros(len(test["time_remaining"]))

    # Global fallback per mode
    k_global = {}
    for ctype in ["heat_increase", "cool_decrease"]:
        mask = train["change_type"] == ctype
        if mask.sum() > 0:
            gaps = np.abs(train["current_gap"][mask])
            times = train["time_remaining"][mask]
            k_global[ctype] = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)

    test_homes = np.unique(test["home_id"])
    n_home_models = 0

    for home in test_homes:
        for ctype in ["heat_increase", "cool_decrease"]:
            train_mask = (train["home_id"] == home) & (train["change_type"] == ctype)
            test_mask = (test["home_id"] == home) & (test["change_type"] == ctype)

            if test_mask.sum() == 0:
                continue

            if train_mask.sum() >= 10:
                gaps = np.abs(train["current_gap"][train_mask])
                times = train["time_remaining"][train_mask]
                k = np.dot(times, gaps) / (np.dot(gaps, gaps) + 1e-8)
                k = np.clip(k, 5, 60)
                n_home_models += 1
            else:
                k = k_global.get(ctype, 20)

            pred[test_mask] = k * np.abs(test["current_gap"][test_mask])

    print(f"    B7: {n_home_models} home-mode models")
    return pred


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Time-to-Target Prediction Baselines")
    print("=" * 70)
    print()

    df = load_data()
    train, test = create_samples(df, train_frac=0.7)

    print()
    print("Running baselines...")
    print("-" * 70)

    results = []

    baselines = [
        ("B1: Global Mean", lambda: B1_global_mean(train, test)),
        ("B2: Gap Proportional", lambda: B2_gap_proportional(train, test)),
        ("B3: Gap + Thermal", lambda: B3_gap_thermal(train, test)),
        ("B4: Mode-Specific", lambda: B4_mode_specific(train, test)),
        ("B5: Per-Home Gap", lambda: B5_per_home_gap(train, test)),
        ("B6: Per-Home Linear", lambda: B6_per_home_linear(train, test)),
        ("B7: Per-Home + Mode", lambda: B7_per_home_mode(train, test)),
    ]

    for name, fn in baselines:
        print(f"  {name}...")
        t0 = time.time()
        pred = fn()
        metrics = compute_metrics(
            pred, test["time_remaining"],
            test["change_type"], test["initial_gap"], test["elapsed_time"]
        )
        elapsed = time.time() - t0
        results.append((name, metrics, elapsed))

    # Print overall results
    print()
    print("=" * 80)
    print("Overall Results (predicting minutes remaining)")
    print("=" * 80)
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'P95':>8} {'Bias':>8}")
    print("-" * 80)
    for name, m, _ in results:
        print(f"{name:<22} {m.mae:>8.1f} {m.rmse:>8.1f} {m.mape:>7.1f}% {m.p95:>8.1f} {m.bias:>+8.1f}")

    # By change type
    print()
    print("=" * 80)
    print("Results by Change Type (MAE in minutes)")
    print("=" * 80)
    print(f"{'Model':<22} {'heat_increase':>14} {'cool_decrease':>14}")
    print("-" * 80)
    for name, m, _ in results:
        h = m.by_type.get("heat_increase")
        c = m.by_type.get("cool_decrease")
        h_mae = f"{h.mae:.1f}" if h else "N/A"
        c_mae = f"{c.mae:.1f}" if c else "N/A"
        print(f"{name:<22} {h_mae:>14} {c_mae:>14}")

    h = results[0][1].by_type.get("heat_increase")
    c = results[0][1].by_type.get("cool_decrease")
    print(f"{'(n samples)':<22} {h.n_samples if h else 0:>14,} {c.n_samples if c else 0:>14,}")

    # By initial gap
    print()
    print("=" * 80)
    print("Results by Initial Gap (MAE in minutes)")
    print("=" * 80)
    gap_names = ["1-2°F", "2-3°F", "3-5°F", ">5°F"]
    header = f"{'Model':<22}" + "".join(f"{g:>12}" for g in gap_names)
    print(header)
    print("-" * 80)
    for name, m, _ in results:
        row = f"{name:<22}"
        for g in gap_names:
            gm = m.by_gap.get(g)
            row += f"{gm.mae:>12.1f}" if gm else f"{'N/A':>12}"
        print(row)

    row = f"{'(n samples)':<22}"
    for g in gap_names:
        gm = results[0][1].by_gap.get(g)
        row += f"{gm.n_samples if gm else 0:>12,}"
    print(row)

    # By episode stage
    print()
    print("=" * 80)
    print("Results by Prediction Time (MAE in minutes)")
    print("=" * 80)
    stage_names = ["start (0-5min)", "early (5-15min)", "mid (15-30min)", "late (>30min)"]
    header = f"{'Model':<22}" + "".join(f"{s:>16}" for s in stage_names)
    print(header)
    print("-" * 80)
    for name, m, _ in results:
        row = f"{name:<22}"
        for s in stage_names:
            sm = m.by_stage.get(s)
            row += f"{sm.mae:>16.1f}" if sm else f"{'N/A':>16}"
        print(row)

    row = f"{'(n samples)':<22}"
    for s in stage_names:
        sm = results[0][1].by_stage.get(s)
        row += f"{sm.n_samples if sm else 0:>16,}"
    print(row)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
