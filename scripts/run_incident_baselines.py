"""
Baseline models for thermal prediction within incidents.

Unlike run_baselines.py which samples arbitrary timesteps, this script
predicts 30 minutes ahead within goal-oriented episodes where we know:
- The episode type (heating, cooling, drift, setpoint)
- The target setpoint
- The initial gap from setpoint
- Whether HVAC is fighting the thermal gradient (is_active)
- Position within the episode (timestep_idx)

Usage: python scripts/run_incident_baselines.py [--incident-type TYPE]
"""

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Suppress sklearn warnings about numerical issues
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
DATA_DIR = Path("data/incidents")
HORIZON = 6  # 30 minutes ahead (6 x 5-min timesteps)
SAMPLE_RATE = 12  # Sample every 12 timesteps within episode (1 hour)


@dataclass
class StratifiedMetrics:
    """Metrics stratified by various dimensions."""
    mae: float = 0.0
    rmse: float = 0.0
    p95: float = 0.0
    bias: float = 0.0
    n_samples: int = 0


@dataclass
class Metrics:
    """Full evaluation metrics."""
    mae: float = 0.0
    rmse: float = 0.0
    max_error: float = 0.0
    p95: float = 0.0
    bias: float = 0.0
    r2: float = 0.0
    by_active: Dict[bool, StratifiedMetrics] = field(default_factory=dict)
    by_gap_bucket: Dict[str, StratifiedMetrics] = field(default_factory=dict)
    by_position: Dict[str, StratifiedMetrics] = field(default_factory=dict)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    is_active: np.ndarray,
    initial_gap: np.ndarray,
    timestep_idx: np.ndarray,
) -> Metrics:
    """Compute evaluation metrics with stratification."""
    errors = pred - target
    abs_errors = np.abs(errors)

    m = Metrics()
    m.mae = abs_errors.mean()
    m.rmse = np.sqrt((errors ** 2).mean())
    m.max_error = abs_errors.max()
    m.p95 = np.percentile(abs_errors, 95)
    m.bias = errors.mean()

    ss_res = (errors ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    m.r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # By is_active
    for active_val in [True, False]:
        mask = is_active == active_val
        if mask.sum() > 0:
            mode_errors = errors[mask]
            mode_abs = abs_errors[mask]
            m.by_active[active_val] = StratifiedMetrics(
                mae=mode_abs.mean(),
                rmse=np.sqrt((mode_errors ** 2).mean()),
                p95=np.percentile(mode_abs, 95),
                bias=mode_errors.mean(),
                n_samples=mask.sum(),
            )

    # By gap bucket
    gap_buckets = np.where(
        initial_gap < 1, "small (<1°F)",
        np.where(initial_gap < 3, "medium (1-3°F)", "large (>3°F)")
    )
    for bucket in ["small (<1°F)", "medium (1-3°F)", "large (>3°F)"]:
        mask = gap_buckets == bucket
        if mask.sum() > 0:
            bucket_errors = errors[mask]
            bucket_abs = abs_errors[mask]
            m.by_gap_bucket[bucket] = StratifiedMetrics(
                mae=bucket_abs.mean(),
                rmse=np.sqrt((bucket_errors ** 2).mean()),
                p95=np.percentile(bucket_abs, 95),
                bias=bucket_errors.mean(),
                n_samples=mask.sum(),
            )

    # By position in episode
    pos_buckets = np.where(
        timestep_idx < 6, "early (0-30min)",
        np.where(timestep_idx < 12, "mid (30-60min)", "late (>60min)")
    )
    for bucket in ["early (0-30min)", "mid (30-60min)", "late (>60min)"]:
        mask = pos_buckets == bucket
        if mask.sum() > 0:
            pos_errors = errors[mask]
            pos_abs = abs_errors[mask]
            m.by_position[bucket] = StratifiedMetrics(
                mae=pos_abs.mean(),
                rmse=np.sqrt((pos_errors ** 2).mean()),
                p95=np.percentile(pos_abs, 95),
                bias=pos_errors.mean(),
                n_samples=mask.sum(),
            )

    return m


def load_incidents(incident_type: str) -> pl.DataFrame:
    """Load incident parquet file."""
    filepath = DATA_DIR / f"{incident_type}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading {incident_type} incidents...")
    t0 = time.time()
    df = pl.read_parquet(filepath)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def create_samples(df: pl.DataFrame, split: str) -> dict:
    """
    Create 30-min prediction samples within episodes.

    For each sample point t, we predict Indoor_AverageTemperature at t+6.
    Only include samples where episode continues for 6+ more timesteps.
    """
    print(f"Creating {split} samples...")
    t0 = time.time()

    split_df = df.filter(pl.col("split") == split)

    # Filter to samples with enough future timesteps in same episode
    # episode_duration - timestep_idx > HORIZON
    split_df = split_df.filter(
        pl.col("episode_duration") - pl.col("timestep_idx") > HORIZON
    )

    # Sample within episodes to reduce dataset size
    # Take every SAMPLE_RATE-th timestep within each episode
    split_df = split_df.filter(
        pl.col("timestep_idx") % SAMPLE_RATE == 0
    )

    # Sort for proper shifting
    split_df = split_df.sort(["incident_id", "timestep_idx"])

    # Get future temperature (target)
    # We need to join with future timestep in same episode
    split_df = split_df.with_columns([
        (pl.col("timestep_idx") + HORIZON).alias("future_idx")
    ])

    # Create lookup for future temps
    future_lookup = df.select([
        "incident_id",
        "timestep_idx",
        pl.col("Indoor_AverageTemperature").alias("target_temp"),
        pl.col("Outdoor_Temperature").alias("future_outdoor"),
    ])

    # Join to get future temperature
    split_df = split_df.join(
        future_lookup,
        left_on=["incident_id", "future_idx"],
        right_on=["incident_id", "timestep_idx"],
        how="inner"
    )

    # Extract hour from timestamp
    split_df = split_df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour")
    ])

    # Add derived features
    split_df = split_df.with_columns([
        # Thermal driving force
        (pl.col("Outdoor_Temperature") - pl.col("Indoor_AverageTemperature")).alias("thermal_drive"),
        # Current gap to target
        (pl.col("target_setpoint") - pl.col("Indoor_AverageTemperature")).alias("current_gap"),
        # Progress: how much of initial gap has been closed (clamped to avoid extreme values)
        (
            (pl.col("Indoor_AverageTemperature") - pl.col("start_indoor_temp")) /
            (pl.col("initial_gap").clip(lower_bound=0.5))  # Avoid div by small numbers
        ).clip(-5.0, 5.0).alias("progress"),
        # Temperature change since episode start
        (pl.col("Indoor_AverageTemperature") - pl.col("start_indoor_temp")).alias("temp_change_so_far"),
        # Time features (cyclical)
        (2 * np.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
        (2 * np.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
    ])

    # Filter out rows with invalid temperatures (0.0 indicates missing)
    split_df = split_df.filter(
        (pl.col("Indoor_AverageTemperature") > 0) &
        (pl.col("Outdoor_Temperature").is_not_null()) &
        (pl.col("target_temp") > 0)
    )

    # Convert to numpy dict
    samples = {}
    columns = [
        "Indoor_AverageTemperature", "Outdoor_Temperature", "future_outdoor",
        "Indoor_HeatSetpoint", "Indoor_CoolSetpoint",
        "target_temp", "state", "hour",
        "thermal_drive", "current_gap", "progress", "temp_change_so_far",
        "hour_sin", "hour_cos",
        # Episode context
        "target_setpoint", "initial_gap", "is_active", "timestep_idx",
        "start_indoor_temp", "start_outdoor_temp", "episode_duration",
        "incident_type",
    ]

    for col in columns:
        if col in split_df.columns:
            arr = split_df[col].to_numpy()
            # Handle string columns
            if arr.dtype == object or col in ["state", "incident_type"]:
                samples[col] = arr
            else:
                arr = arr.astype(np.float64)
                # Replace inf with 0
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                samples[col] = arr

    # Rename for consistency with existing baseline code
    samples["indoor_temp"] = samples["Indoor_AverageTemperature"]
    samples["outdoor_temp"] = samples["Outdoor_Temperature"]
    samples["heat_sp"] = samples["Indoor_HeatSetpoint"]
    samples["cool_sp"] = samples["Indoor_CoolSetpoint"]
    samples["target"] = samples["target_temp"]

    print(f"  {len(samples['target']):,} samples in {time.time()-t0:.1f}s")

    # Print stratification stats
    n = len(samples["target"])
    n_active = (samples["is_active"] == True).sum()
    print(f"  Active: {n_active:,} ({100*n_active/n:.1f}%) | Passive: {n-n_active:,} ({100*(n-n_active)/n:.1f}%)")

    gaps = samples["initial_gap"]
    small = (gaps < 1).sum()
    med = ((gaps >= 1) & (gaps < 3)).sum()
    large = (gaps >= 3).sum()
    print(f"  Gap buckets: small={100*small/n:.1f}% | medium={100*med/n:.1f}% | large={100*large/n:.1f}%")

    return samples


# =============================================================================
# Baseline Models
# =============================================================================

def B1_persistence(test: dict) -> np.ndarray:
    """B1: Predict current temperature stays the same."""
    return test["indoor_temp"].copy()


def B2_drift_to_target(train: dict, test: dict) -> np.ndarray:
    """B2: Linear drift toward episode target setpoint."""
    # Learn: delta = k * current_gap
    X_train = train["current_gap"].reshape(-1, 1)
    y_train = train["target"] - train["indoor_temp"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = test["current_gap"].reshape(-1, 1)
    delta = model.predict(X_test)
    return test["indoor_temp"] + delta


def B3_gap_proportional(train: dict, test: dict) -> np.ndarray:
    """B3: Drift rate proportional to initial gap (larger gap = faster change)."""
    # Learn: delta = k * current_gap + b * initial_gap
    X_train = np.column_stack([
        train["current_gap"],
        train["initial_gap"],
    ])
    y_train = train["target"] - train["indoor_temp"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = np.column_stack([
        test["current_gap"],
        test["initial_gap"],
    ])
    delta = model.predict(X_test)
    return test["indoor_temp"] + delta


def B4_episode_context(train: dict, test: dict) -> np.ndarray:
    """B4: Use episode context features."""
    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["current_gap"],
            s["initial_gap"],
            s["timestep_idx"],
            s["temp_change_so_far"],
        ])

    X_train = features(train)
    X_test = features(test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, train["target"])
    return model.predict(X_test_scaled)


def B5_thermal_plus_episode(train: dict, test: dict) -> np.ndarray:
    """B5: Combine thermal features with episode context."""
    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["current_gap"],
            s["initial_gap"],
            s["timestep_idx"],
            s["temp_change_so_far"],
        ])

    X_train = features(train)
    X_test = features(test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, train["target"])
    return model.predict(X_test_scaled)


def B6_active_split(train: dict, test: dict) -> np.ndarray:
    """B6: Separate models for active vs passive episodes."""
    pred = np.zeros(len(test["target"]))

    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["current_gap"],
            s["initial_gap"],
            s["timestep_idx"],
            s["temp_change_so_far"],
        ])

    for active_val in [True, False]:
        train_mask = train["is_active"] == active_val
        test_mask = test["is_active"] == active_val

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            pred[test_mask] = test["indoor_temp"][test_mask]
            continue

        train_sub = {k: v[train_mask] for k, v in train.items()}
        test_sub = {k: v[test_mask] for k, v in test.items()}

        X_train = features(train_sub)
        X_test = features(test_sub)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, train_sub["target"])
        pred[test_mask] = model.predict(X_test_scaled)

    return pred


def B7_full_features(train: dict, test: dict) -> np.ndarray:
    """B7: Full feature set including time."""
    def features(s):
        return np.column_stack([
            s["indoor_temp"],
            s["outdoor_temp"],
            s["thermal_drive"],
            s["heat_sp"],
            s["cool_sp"],
            s["current_gap"],
            s["initial_gap"],
            s["timestep_idx"],
            s["temp_change_so_far"],
            s["progress"],
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


def B8_exponential_decay(train: dict, test: dict) -> np.ndarray:
    """B8: Exponential decay toward target (Newton's law approximation)."""
    # For heating: T(t+dt) = target - (target - T(t)) * exp(-dt/tau)
    # Linearized: delta = (1 - exp(-dt/tau)) * current_gap
    # Learn the effective (1 - exp(-dt/tau)) coefficient

    # This is equivalent to B2 but we interpret it as exponential decay
    # For better fit, we can use gap * some decay factor based on timestep

    # Learn: delta = alpha * current_gap * decay_factor
    # where decay_factor accounts for how much of episode remains

    # Simplified: just use current_gap weighted by position
    remaining_frac = 1 - (train["timestep_idx"] / (train["episode_duration"] + 1))
    X_train = (train["current_gap"] * remaining_frac).reshape(-1, 1)
    y_train = train["target"] - train["indoor_temp"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    remaining_frac_test = 1 - (test["timestep_idx"] / (test["episode_duration"] + 1))
    X_test = (test["current_gap"] * remaining_frac_test).reshape(-1, 1)
    delta = model.predict(X_test)
    return test["indoor_temp"] + delta


# =============================================================================
# Main
# =============================================================================

def print_results(results: list, test: dict):
    """Print formatted results tables."""
    n = len(test["target"])

    # Overall results
    print()
    print("=" * 90)
    print("Overall Results")
    print("=" * 90)
    print(f"{'Model':<28} {'MAE':>7} {'RMSE':>7} {'P95':>7} {'Bias':>8} {'R²':>7}")
    print("-" * 90)
    for name, m, _ in results:
        print(f"{name:<28} {m.mae:>7.3f} {m.rmse:>7.3f} {m.p95:>7.3f} {m.bias:>+8.3f} {m.r2:>7.3f}")

    # By is_active
    print()
    print("=" * 90)
    print("By Active Status (HVAC fighting thermal gradient)")
    print("=" * 90)
    active_n = results[0][1].by_active.get(True, StratifiedMetrics()).n_samples
    passive_n = results[0][1].by_active.get(False, StratifiedMetrics()).n_samples
    print(f"Active: {active_n:,} samples ({100*active_n/n:.1f}%) | Passive: {passive_n:,} samples ({100*passive_n/n:.1f}%)")
    print()
    print(f"{'Model':<28} {'Active MAE':>12} {'Passive MAE':>12} {'Active Bias':>12} {'Passive Bias':>12}")
    print("-" * 90)
    for name, m, _ in results:
        a = m.by_active.get(True, StratifiedMetrics())
        p = m.by_active.get(False, StratifiedMetrics())
        print(f"{name:<28} {a.mae:>12.3f} {p.mae:>12.3f} {a.bias:>+12.3f} {p.bias:>+12.3f}")

    # By gap bucket
    print()
    print("=" * 90)
    print("By Initial Gap Size")
    print("=" * 90)
    for bucket in ["small (<1°F)", "medium (1-3°F)", "large (>3°F)"]:
        bn = results[0][1].by_gap_bucket.get(bucket, StratifiedMetrics()).n_samples
        print(f"  {bucket}: {bn:,} samples ({100*bn/n:.1f}%)")
    print()
    print(f"{'Model':<28} {'Small':>10} {'Medium':>10} {'Large':>10}")
    print("-" * 90)
    for name, m, _ in results:
        s = m.by_gap_bucket.get("small (<1°F)", StratifiedMetrics())
        med = m.by_gap_bucket.get("medium (1-3°F)", StratifiedMetrics())
        l = m.by_gap_bucket.get("large (>3°F)", StratifiedMetrics())
        print(f"{name:<28} {s.mae:>10.3f} {med.mae:>10.3f} {l.mae:>10.3f}")

    # By position in episode
    print()
    print("=" * 90)
    print("By Position in Episode")
    print("=" * 90)
    for bucket in ["early (0-30min)", "mid (30-60min)", "late (>60min)"]:
        bn = results[0][1].by_position.get(bucket, StratifiedMetrics()).n_samples
        print(f"  {bucket}: {bn:,} samples ({100*bn/n:.1f}%)")
    print()
    print(f"{'Model':<28} {'Early':>10} {'Mid':>10} {'Late':>10}")
    print("-" * 90)
    for name, m, _ in results:
        e = m.by_position.get("early (0-30min)", StratifiedMetrics())
        mid = m.by_position.get("mid (30-60min)", StratifiedMetrics())
        l = m.by_position.get("late (>60min)", StratifiedMetrics())
        print(f"{name:<28} {e.mae:>10.3f} {mid.mae:>10.3f} {l.mae:>10.3f}")

    print("=" * 90)


def main(incident_type: str = "heating"):
    print("=" * 70)
    print(f"Incident-Based Thermal Prediction Baselines")
    print("=" * 70)
    print(f"Incident type: {incident_type}")
    print(f"Horizon: {HORIZON} steps ({HORIZON * 5} min)")
    print(f"Sample rate: every {SAMPLE_RATE} steps within episode ({SAMPLE_RATE * 5} min)")
    print()

    # Load data
    df = load_incidents(incident_type)

    # Create samples
    train = create_samples(df, "train")
    test = create_samples(df, "test")

    print()
    print("Running baselines...")
    print("-" * 70)

    results = []

    baselines = [
        ("B1: Persistence", lambda: B1_persistence(test)),
        ("B2: Drift to Target", lambda: B2_drift_to_target(train, test)),
        ("B3: Gap Proportional", lambda: B3_gap_proportional(train, test)),
        ("B4: Episode Context", lambda: B4_episode_context(train, test)),
        ("B5: Thermal + Episode", lambda: B5_thermal_plus_episode(train, test)),
        ("B6: Active/Passive Split", lambda: B6_active_split(train, test)),
        ("B7: Full Features", lambda: B7_full_features(train, test)),
        ("B8: Exponential Decay", lambda: B8_exponential_decay(train, test)),
    ]

    for name, fn in tqdm(baselines, desc="Baselines"):
        t0 = time.time()
        pred = fn()
        metrics = compute_metrics(
            pred,
            test["target"],
            test["is_active"],
            test["initial_gap"],
            test["timestep_idx"],
        )
        elapsed = time.time() - t0
        results.append((name, metrics, elapsed))

    print_results(results, test)

    # Summary
    print()
    print("Key Findings:")
    best_model = min(results, key=lambda x: x[1].mae)
    print(f"  Best overall: {best_model[0]} with {best_model[1].mae:.3f}°F MAE")

    persistence_mae = results[0][1].mae
    for name, m, _ in results[1:]:
        if m.mae < persistence_mae:
            improvement = 100 * (persistence_mae - m.mae) / persistence_mae
            print(f"  {name} beats persistence by {improvement:.1f}%")


def run_all_types():
    """Run baselines for all incident types."""
    for incident_type in ["heating", "cooling", "drift", "setpoint"]:
        filepath = DATA_DIR / f"{incident_type}.parquet"
        if filepath.exists():
            print("\n" + "=" * 70)
            print(f"INCIDENT TYPE: {incident_type.upper()}")
            print("=" * 70 + "\n")
            main(incident_type)
        else:
            print(f"Skipping {incident_type} (file not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline models on incident dataset"
    )
    parser.add_argument(
        "--incident-type",
        choices=["heating", "cooling", "drift", "setpoint", "all"],
        default="heating",
        help="Which incident type to run baselines on (default: heating)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=12,
        help="Sample every N timesteps within episodes (default: 12 = 1 hour)"
    )
    args = parser.parse_args()

    SAMPLE_RATE = args.sample_rate

    if args.incident_type == "all":
        run_all_types()
    else:
        main(args.incident_type)
