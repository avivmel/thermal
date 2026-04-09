"""
Extract drift episodes from Ecobee data.

Drift = HVAC off, temperature drifting toward outdoor ambient.
Used for training passive thermal models for MPC.

Episode definition:
- Start: All available runtime columns = 0, |indoor - outdoor| > 3F
- End: Any runtime > 0, setpoint changes > 1F, or data gap
- Label: first passage time to the relevant comfort boundary
- Keep only episodes with an observed boundary crossing
"""

import argparse
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("ecobee_processed_dataset/extracted/clean_data")
OUTPUT_FILE = Path("data/drift_episodes.parquet")
THERMAL_DATASET = Path("data/thermal_dataset.csv")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# All possible runtime columns - we check which exist per dataset
RUNTIME_COLUMNS = [
    "HeatingEquipmentStage1_RunTime",
    "HeatingEquipmentStage2_RunTime",
    "HeatingEquipmentStage3_RunTime",
    "CoolingEquipmentStage1_RunTime",
    "CoolingEquipmentStage2_RunTime",
    "HeatPumpsStage1_RunTime",
    "HeatPumpsStage2_RunTime",
]

# Columns to keep in output
KEEP_COLUMNS = [
    "Indoor_AverageTemperature",
    "Outdoor_Temperature",
    "Indoor_HeatSetpoint",
    "Indoor_CoolSetpoint",
    "Indoor_Humidity",
    "Outdoor_Humidity",
]

MIN_DELTA_F = 3.0
MIN_TIME_TO_BOUNDARY_MIN = 15
MAX_SETPOINT_CHANGE = 1.0  # End episode if setpoint changes more than this
MAX_GAP_STEPS = 2       # End episode if data gap > 2 timesteps (10 min)
SLOPE_WINDOWS = {
    "15m": 3,
    "30m": 6,
}


def is_missing_temp(value: float) -> bool:
    return value == 0 or np.isnan(value)


def has_gap(times: np.ndarray, prev_idx: int, cur_idx: int) -> bool:
    prev_ts = pd.Timestamp(times[prev_idx])
    cur_ts = pd.Timestamp(times[cur_idx])
    gap_minutes = (cur_ts - prev_ts).total_seconds() / 60.0
    return gap_minutes > MAX_GAP_STEPS * 5


def is_hvac_running(runtime_arrays: list, idx: int) -> bool:
    for rt in runtime_arrays:
        val = rt[idx]
        if not np.isnan(val) and val > 0:
            return True
    return False


def compute_recent_slope(values: np.ndarray, times: np.ndarray, end_idx: int, window_steps: int) -> float:
    """Compute average slope in F/hour over the preceding window ending at end_idx."""
    start_idx = end_idx - window_steps
    if start_idx < 0:
        return np.nan

    for idx in range(start_idx, end_idx + 1):
        if is_missing_temp(values[idx]):
            return np.nan
        if idx > start_idx and has_gap(times, idx - 1, idx):
            return np.nan

    elapsed_hours = (pd.Timestamp(times[end_idx]) - pd.Timestamp(times[start_idx])).total_seconds() / 3600.0
    if elapsed_hours <= 0:
        return np.nan
    return (values[end_idx] - values[start_idx]) / elapsed_hours


def compute_time_since_hvac_off(times: np.ndarray, indoor: np.ndarray, outdoor: np.ndarray,
                                runtime_arrays: list, start_idx: int) -> float:
    """Minutes since HVAC last turned off, counting the contiguous off segment before the episode start."""
    idx = start_idx
    while idx > 0:
        prev_idx = idx - 1
        if has_gap(times, prev_idx, idx):
            break
        if is_missing_temp(indoor[prev_idx]) or is_missing_temp(outdoor[prev_idx]):
            break
        if is_hvac_running(runtime_arrays, prev_idx):
            break
        idx = prev_idx

    return (pd.Timestamp(times[start_idx]) - pd.Timestamp(times[idx])).total_seconds() / 60.0


def load_home_splits() -> dict:
    """Load home-to-split mapping from thermal_dataset.csv."""
    print("Loading home splits...")
    splits = {}
    for chunk in pd.read_csv(THERMAL_DATASET, usecols=["home_id", "split"], chunksize=1_000_000):
        for home_id, split in chunk.drop_duplicates("home_id").set_index("home_id")["split"].items():
            if home_id not in splits:
                splits[home_id] = split
        if len(splits) >= 971:
            break
    print(f"  Loaded {len(splits)} homes")
    return splits


def find_available_runtimes(ds: xr.Dataset, home_idx: int, all_runtime_data: dict) -> list:
    """Find which runtime columns have non-zero data for a home."""
    available = []
    for col in RUNTIME_COLUMNS:
        if col not in all_runtime_data:
            continue
        vals = all_runtime_data[col][home_idx]
        # Column is "available" if it has any non-zero, non-NaN values
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0 and np.any(valid > 0):
            available.append(col)
    return available


def extract_drift_episodes(
    home_id: str,
    times: np.ndarray,
    data: dict,
    runtime_cols: list,
    state: str,
    split: str,
) -> tuple[list, dict]:
    """
    Extract drift episodes for one home.

    A drift episode is a contiguous HVAC-off period where:
    - All available HVAC runtimes are 0
    - |indoor - outdoor| > MIN_DELTA_F at start
    - The indoor temperature reaches the relevant comfort boundary
    """
    indoor = data["Indoor_AverageTemperature"]
    outdoor = data["Outdoor_Temperature"]
    heat_sp = data["Indoor_HeatSetpoint"]
    cool_sp = data["Indoor_CoolSetpoint"]

    # Stack available runtime columns
    runtime_arrays = [data[col] for col in runtime_cols]

    n = len(times)
    episodes = []
    stats = {
        "candidate_intervals": 0,
        "crossed_intervals": 0,
        "dropped_no_crossing": 0,
    }
    i = 0

    while i < n:
        # Skip missing data
        if is_missing_temp(indoor[i]) or is_missing_temp(outdoor[i]):
            i += 1
            continue

        # Check: all runtimes must be 0
        if is_hvac_running(runtime_arrays, i):
            i += 1
            continue

        # Check: sufficient thermal delta
        delta = indoor[i] - outdoor[i]
        if abs(delta) < MIN_DELTA_F:
            i += 1
            continue

        # Start of a potential HVAC-off interval.
        stats["candidate_intervals"] += 1
        start_idx = i
        start_temp = indoor[i]
        start_outdoor = outdoor[i]
        start_heat_sp = heat_sp[i] if not np.isnan(heat_sp[i]) else 0
        start_cool_sp = cool_sp[i] if not np.isnan(cool_sp[i]) else 0

        # Collect timesteps until episode ends
        j = i + 1
        while j < n:
            if has_gap(times, j - 1, j):
                break

            # Data gap check
            if is_missing_temp(indoor[j]) or is_missing_temp(outdoor[j]):
                break

            # Runtime check - any equipment turns on?
            if is_hvac_running(runtime_arrays, j):
                break

            # Setpoint change check
            cur_heat_sp = heat_sp[j] if not np.isnan(heat_sp[j]) else 0
            cur_cool_sp = cool_sp[j] if not np.isnan(cool_sp[j]) else 0
            if abs(cur_heat_sp - start_heat_sp) > MAX_SETPOINT_CHANGE:
                break
            if abs(cur_cool_sp - start_cool_sp) > MAX_SETPOINT_CHANGE:
                break

            j += 1

        end_idx = j  # exclusive
        initial_delta = start_temp - start_outdoor

        # Match the reformulated task:
        # - cooling_drift -> time until indoor reaches the cooling boundary
        # - warming_drift -> time until indoor reaches the heating boundary
        if initial_delta < 0:
            drift_direction = "cooling_drift"
            target_boundary = "cool_setpoint"
            boundary_temp = start_cool_sp
            crossing_mask = indoor[start_idx:end_idx] >= boundary_temp
        else:
            drift_direction = "warming_drift"
            target_boundary = "heat_setpoint"
            boundary_temp = start_heat_sp
            crossing_mask = indoor[start_idx:end_idx] <= boundary_temp

        if boundary_temp == 0 or np.isnan(boundary_temp):
            i = end_idx
            continue

        signed_boundary_gap = boundary_temp - start_temp
        distance_to_boundary = abs(signed_boundary_gap)

        # The episode should start inside the comfort band and move toward the boundary.
        if distance_to_boundary <= 0:
            i = end_idx
            continue

        crossing_indices = np.flatnonzero(crossing_mask)
        if len(crossing_indices) == 0:
            stats["dropped_no_crossing"] += 1
            i = end_idx
            continue

        crossing_timestep_idx = int(crossing_indices[0])
        time_to_boundary_min = crossing_timestep_idx * 5

        if time_to_boundary_min < MIN_TIME_TO_BOUNDARY_MIN:
            i = end_idx
            continue

        crossing_end_idx = start_idx + crossing_timestep_idx + 1
        stats["crossed_intervals"] += 1

        indoor_slope_15m = compute_recent_slope(indoor, times, start_idx, SLOPE_WINDOWS["15m"])
        indoor_slope_30m = compute_recent_slope(indoor, times, start_idx, SLOPE_WINDOWS["30m"])
        outdoor_slope_15m = compute_recent_slope(outdoor, times, start_idx, SLOPE_WINDOWS["15m"])
        outdoor_slope_30m = compute_recent_slope(outdoor, times, start_idx, SLOPE_WINDOWS["30m"])
        time_since_hvac_off_min = compute_time_since_hvac_off(
            times, indoor, outdoor, runtime_arrays, start_idx
        )

        # Build episode rows
        ep_id = f"{home_id}_drift_{pd.Timestamp(times[start_idx]).isoformat()}"
        episode_rows = []

        for k in range(start_idx, crossing_end_idx):
            row = {
                "home_id": home_id,
                "state": state,
                "split": split,
                "episode_id": ep_id,
                "timestamp": pd.Timestamp(times[k]),
                "timestep_idx": k - start_idx,
                "drift_direction": drift_direction,
                "start_temp": start_temp,
                "start_outdoor": start_outdoor,
                "initial_delta": initial_delta,
                "target_boundary": target_boundary,
                "boundary_temp": boundary_temp,
                "crossed_boundary": True,
                "crossing_timestep_idx": crossing_timestep_idx,
                "time_to_boundary_min": time_to_boundary_min,
                "distance_to_boundary": distance_to_boundary,
                "signed_boundary_gap": signed_boundary_gap,
                "recent_indoor_slope_15m": indoor_slope_15m,
                "recent_indoor_slope_30m": indoor_slope_30m,
                "recent_outdoor_slope_15m": outdoor_slope_15m,
                "recent_outdoor_slope_30m": outdoor_slope_30m,
                "time_since_hvac_off_min": time_since_hvac_off_min,
            }
            for col in KEEP_COLUMNS:
                if col in data:
                    row[col] = data[col][k]
            episode_rows.append(row)

        episodes.append(pd.DataFrame(episode_rows))
        i = end_idx

    return episodes, stats


def process_month(month: str, home_splits: dict, max_homes: int = None) -> pd.DataFrame:
    """Process one month of data."""
    filepath = DATA_DIR / f"{month}_clean.nc"
    ds = xr.open_dataset(filepath)

    # Get home states
    states = dict(zip(ds.id.values, ds["State"].isel(time=0).values))

    # Filter to valid homes
    valid_homes = [h for h in ds.id.values if h in home_splits and states.get(h, "")]
    if max_homes:
        valid_homes = valid_homes[:max_homes]

    ds = ds.sel(id=valid_homes)
    home_ids = ds.id.values
    times = ds.time.values

    # Find which runtime columns exist in this dataset
    available_runtime_cols = [c for c in RUNTIME_COLUMNS if c in ds]

    # Load data
    all_data = {}
    for col in KEEP_COLUMNS:
        if col in ds:
            all_data[col] = ds[col].values
    for col in available_runtime_cols:
        all_data[col] = ds[col].values

    all_episodes = []
    summary = {
        "candidate_intervals": 0,
        "crossed_intervals": 0,
        "dropped_no_crossing": 0,
    }

    for idx, home_id in enumerate(tqdm(home_ids, desc=f"  {month}", leave=False)):
        home_data = {col: arr[idx] for col, arr in all_data.items()}

        # Find which runtime columns this home actually uses
        home_rt_cols = find_available_runtimes(ds, idx, all_data)
        if not home_rt_cols:
            # No runtime data at all - skip home (can't determine HVAC state)
            continue

        episodes, stats = extract_drift_episodes(
            home_id=home_id,
            times=times,
            data=home_data,
            runtime_cols=home_rt_cols,
            state=states[home_id],
            split=home_splits[home_id],
        )
        all_episodes.extend(episodes)
        for key, value in stats.items():
            summary[key] += value

    ds.close()

    if all_episodes:
        return pd.concat(all_episodes, ignore_index=True), summary
    return pd.DataFrame(), summary


def main(test_mode: bool = False, n_homes: int = 10):
    print("=" * 60)
    print("Extracting Drift Episodes (HVAC off)")
    if test_mode:
        print(f"  TEST MODE: {n_homes} homes, January only")
    print("=" * 60)

    home_splits = load_home_splits()

    months = ["Jan"] if test_mode else MONTHS

    all_dfs = []
    aggregate_summary = {
        "candidate_intervals": 0,
        "crossed_intervals": 0,
        "dropped_no_crossing": 0,
    }
    for month in tqdm(months, desc="Months"):
        df, summary = process_month(month, home_splits, max_homes=n_homes if test_mode else None)
        for key, value in summary.items():
            aggregate_summary[key] += value
        if len(df) > 0:
            all_dfs.append(df)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result.to_parquet(OUTPUT_FILE, index=False)

        n_episodes = result["episode_id"].nunique()
        print(f"\nOutput: {OUTPUT_FILE}")
        print(f"  Rows: {len(result):,}")
        print(f"  Episodes: {n_episodes:,}")
        print(f"  By direction: {result.groupby('drift_direction').episode_id.nunique().to_dict()}")
        print(f"  By boundary: {result.groupby('target_boundary').episode_id.nunique().to_dict()}")
        print(f"  By split: {result.groupby('split').episode_id.nunique().to_dict()}")
        if aggregate_summary["candidate_intervals"] > 0:
            crossed_frac = aggregate_summary["crossed_intervals"] / aggregate_summary["candidate_intervals"]
            print(f"\n  Candidate HVAC-off intervals: {aggregate_summary['candidate_intervals']:,}")
            print(f"  Observed boundary crossings: {aggregate_summary['crossed_intervals']:,} ({crossed_frac:.1%})")
            print(f"  Dropped without crossing: {aggregate_summary['dropped_no_crossing']:,}")

        time_to_boundary = result.groupby("episode_id")["time_to_boundary_min"].first()
        print(f"\n  Time-to-boundary stats (minutes):")
        print(f"    Mean: {time_to_boundary.mean():.1f}, Median: {time_to_boundary.median():.1f}")
        print(f"    P10: {time_to_boundary.quantile(0.1):.0f}, P90: {time_to_boundary.quantile(0.9):.0f}")

        boundary_gap = result.groupby("episode_id")["distance_to_boundary"].first()
        print(f"\n  Distance-to-boundary stats (F):")
        print(f"    Mean: {boundary_gap.mean():.1f}, Median: {boundary_gap.median():.1f}")
    else:
        print("No drift episodes found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-homes", type=int, default=10)
    args = parser.parse_args()

    main(test_mode=args.test, n_homes=args.n_homes)
