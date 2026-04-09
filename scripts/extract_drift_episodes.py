"""
Extract drift episodes from Ecobee data.

Drift = HVAC off, temperature drifting toward outdoor ambient.
Used for training drift thermal models for MPC.

Episode definition:
- Start: All available runtime columns = 0, |indoor - outdoor| > 3F
- End: Any runtime > 0, setpoint changes > 1F, or data gap
- Min duration: 15 min (3 timesteps at 5-min resolution)
- Filter: |end_temp - start_temp| > 0.5F (need actual temp change)
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

MIN_DELTA_F = 3.0       # Minimum |indoor - outdoor| to start episode
MIN_DURATION_STEPS = 3  # 15 minutes
MIN_TEMP_CHANGE = 0.5   # Must see actual temp movement
MAX_SETPOINT_CHANGE = 1.0  # End episode if setpoint changes more than this
MAX_GAP_STEPS = 2       # End episode if data gap > 2 timesteps (10 min)


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
) -> list:
    """
    Extract drift episodes for one home.

    A drift episode is a contiguous period where:
    - All available HVAC runtimes are 0
    - |indoor - outdoor| > MIN_DELTA_F at start
    - Temperature actually changes during the episode
    """
    indoor = data["Indoor_AverageTemperature"]
    outdoor = data["Outdoor_Temperature"]
    heat_sp = data["Indoor_HeatSetpoint"]
    cool_sp = data["Indoor_CoolSetpoint"]

    # Stack available runtime columns
    runtime_arrays = [data[col] for col in runtime_cols]

    n = len(times)
    episodes = []
    i = 0

    while i < n:
        # Skip missing data
        if indoor[i] == 0 or outdoor[i] == 0 or np.isnan(indoor[i]) or np.isnan(outdoor[i]):
            i += 1
            continue

        # Check: all runtimes must be 0
        any_running = False
        for rt in runtime_arrays:
            val = rt[i]
            if not np.isnan(val) and val > 0:
                any_running = True
                break
        if any_running:
            i += 1
            continue

        # Check: sufficient thermal delta
        delta = indoor[i] - outdoor[i]
        if abs(delta) < MIN_DELTA_F:
            i += 1
            continue

        # Start of a potential drift episode
        start_idx = i
        start_temp = indoor[i]
        start_outdoor = outdoor[i]
        start_heat_sp = heat_sp[i] if not np.isnan(heat_sp[i]) else 0
        start_cool_sp = cool_sp[i] if not np.isnan(cool_sp[i]) else 0

        # Collect timesteps until episode ends
        j = i + 1
        while j < n:
            # Data gap check
            if indoor[j] == 0 or outdoor[j] == 0 or np.isnan(indoor[j]) or np.isnan(outdoor[j]):
                break

            # Runtime check - any equipment turns on?
            equip_on = False
            for rt in runtime_arrays:
                val = rt[j]
                if not np.isnan(val) and val > 0:
                    equip_on = True
                    break
            if equip_on:
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
        n_steps = end_idx - start_idx

        # Duration filter
        if n_steps < MIN_DURATION_STEPS:
            i = end_idx
            continue

        # Temperature change filter
        end_temp = indoor[end_idx - 1]
        if abs(end_temp - start_temp) < MIN_TEMP_CHANGE:
            i = end_idx
            continue

        # Determine drift direction
        initial_delta = start_temp - start_outdoor
        if initial_delta > 0:
            # Indoor warmer than outdoor -> cooling drift (temp falls toward outdoor)
            drift_direction = "cooling_drift"
        else:
            # Indoor cooler than outdoor -> warming drift (temp rises toward outdoor)
            drift_direction = "warming_drift"

        # Build episode rows
        ep_id = f"{home_id}_drift_{pd.Timestamp(times[start_idx]).isoformat()}"
        episode_rows = []

        for k in range(start_idx, end_idx):
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
            }
            for col in KEEP_COLUMNS:
                if col in data:
                    row[col] = data[col][k]
            episode_rows.append(row)

        episodes.append(pd.DataFrame(episode_rows))
        i = end_idx

    return episodes


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

    for idx, home_id in enumerate(tqdm(home_ids, desc=f"  {month}", leave=False)):
        home_data = {col: arr[idx] for col, arr in all_data.items()}

        # Find which runtime columns this home actually uses
        home_rt_cols = find_available_runtimes(ds, idx, all_data)
        if not home_rt_cols:
            # No runtime data at all - skip home (can't determine HVAC state)
            continue

        episodes = extract_drift_episodes(
            home_id=home_id,
            times=times,
            data=home_data,
            runtime_cols=home_rt_cols,
            state=states[home_id],
            split=home_splits[home_id],
        )
        all_episodes.extend(episodes)

    ds.close()

    if all_episodes:
        return pd.concat(all_episodes, ignore_index=True)
    return pd.DataFrame()


def main(test_mode: bool = False, n_homes: int = 10):
    print("=" * 60)
    print("Extracting Drift Episodes (HVAC off)")
    if test_mode:
        print(f"  TEST MODE: {n_homes} homes, January only")
    print("=" * 60)

    home_splits = load_home_splits()

    months = ["Jan"] if test_mode else MONTHS

    all_dfs = []
    for month in tqdm(months, desc="Months"):
        df = process_month(month, home_splits, max_homes=n_homes if test_mode else None)
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
        print(f"  By split: {result.groupby('split').episode_id.nunique().to_dict()}")

        # Duration stats
        dur = result.groupby("episode_id").size() * 5  # minutes
        print(f"\n  Duration stats (minutes):")
        print(f"    Mean: {dur.mean():.1f}, Median: {dur.median():.1f}")
        print(f"    P10: {dur.quantile(0.1):.0f}, P90: {dur.quantile(0.9):.0f}")

        # Temperature change stats
        ep_temps = result.groupby("episode_id").agg(
            start=("Indoor_AverageTemperature", "first"),
            end=("Indoor_AverageTemperature", "last"),
        )
        ep_temps["change"] = (ep_temps["end"] - ep_temps["start"]).abs()
        print(f"\n  Temp change stats (F):")
        print(f"    Mean: {ep_temps['change'].mean():.1f}, Median: {ep_temps['change'].median():.1f}")
    else:
        print("No drift episodes found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-homes", type=int, default=10)
    args = parser.parse_args()

    main(test_mode=args.test, n_homes=args.n_homes)
