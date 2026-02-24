"""
Extract setpoint change response episodes from Ecobee data.

Finds times where setpoint changed by >1°F and saves all rows
until indoor temp reaches the new setpoint value.
"""

import argparse
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("ecobee_processed_dataset/extracted/clean_data")
OUTPUT_FILE = Path("data/setpoint_responses.parquet")
THERMAL_DATASET = Path("data/thermal_dataset.csv")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Columns to keep
COLUMNS = [
    "Indoor_AverageTemperature",
    "Outdoor_Temperature",
    "Indoor_HeatSetpoint",
    "Indoor_CoolSetpoint",
    "Indoor_Humidity",
    "Outdoor_Humidity",
    "HeatingEquipmentStage1_RunTime",
    "CoolingEquipmentStage1_RunTime",
    "Fan_RunTime",
]


def load_home_splits() -> dict:
    """Load home-to-split mapping."""
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


def extract_setpoint_responses(
    home_id: str,
    times: np.ndarray,
    data: dict,
    state: str,
    split: str,
    min_change: float = 1.0
) -> list:
    """
    Extract setpoint change response episodes for one home.

    Args:
        home_id: Home identifier
        times: Array of timestamps
        data: Dict of column -> values array
        state: Home state code
        split: train/val/test
        min_change: Minimum setpoint change to detect (°F)

    Returns:
        List of episode DataFrames
    """
    indoor = data["Indoor_AverageTemperature"]
    heat_sp = data["Indoor_HeatSetpoint"]
    cool_sp = data["Indoor_CoolSetpoint"]

    episodes = []
    n = len(times)
    i = 1  # Start at 1 to compare with previous

    while i < n:
        # Skip missing data
        if indoor[i] == 0 or heat_sp[i] == 0 or cool_sp[i] == 0:
            i += 1
            continue
        if indoor[i-1] == 0 or heat_sp[i-1] == 0 or cool_sp[i-1] == 0:
            i += 1
            continue

        # Detect setpoint change
        heat_change = heat_sp[i] - heat_sp[i-1]
        cool_change = cool_sp[i] - cool_sp[i-1]

        change_type = None
        target = None

        # Heat increase: setpoint goes up, temp needs to rise to meet it
        # Only meaningful if current temp is BELOW new setpoint
        if heat_change > min_change and indoor[i] < heat_sp[i]:
            change_type = "heat_increase"
            target = heat_sp[i]
        # Cool decrease: setpoint goes down, temp needs to fall to meet it
        # Only meaningful if current temp is ABOVE new setpoint
        elif cool_change < -min_change and indoor[i] > cool_sp[i]:
            change_type = "cool_decrease"
            target = cool_sp[i]

        if change_type is None:
            i += 1
            continue

        # Found a setpoint change - collect rows until target reached
        start_idx = i
        start_temp = indoor[i]
        start_time = times[i]

        # Collect rows until temp reaches target (or data ends/gaps)
        episode_rows = []
        j = i

        while j < n:
            # Check for missing data
            if indoor[j] == 0 or heat_sp[j] == 0 or cool_sp[j] == 0:
                break

            # Check if target reached
            reached = False
            if change_type == "heat_increase" and indoor[j] >= target:
                reached = True
            elif change_type == "cool_decrease" and indoor[j] <= target:
                reached = True

            # Build row
            row = {
                "home_id": home_id,
                "state": state,
                "split": split,
                "timestamp": pd.Timestamp(times[j]),
                "episode_id": f"{home_id}_{change_type}_{pd.Timestamp(start_time).isoformat()}",
                "change_type": change_type,
                "target_setpoint": target,
                "start_temp": start_temp,
                "initial_gap": abs(target - start_temp),
                "timestep_idx": j - start_idx,
            }

            # Add data columns
            for col in COLUMNS:
                if col in data:
                    row[col] = data[col][j]

            episode_rows.append(row)

            if reached:
                break

            j += 1

        if episode_rows:
            episodes.append(pd.DataFrame(episode_rows))

        # Move past this episode
        i = j + 1

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

    # Load data
    all_data = {col: ds[col].values for col in COLUMNS if col in ds}

    all_episodes = []

    for idx, home_id in enumerate(tqdm(home_ids, desc=f"  {month}", leave=False)):
        home_data = {col: arr[idx] for col, arr in all_data.items()}

        episodes = extract_setpoint_responses(
            home_id=home_id,
            times=times,
            data=home_data,
            state=states[home_id],
            split=home_splits[home_id]
        )
        all_episodes.extend(episodes)

    ds.close()

    if all_episodes:
        return pd.concat(all_episodes, ignore_index=True)
    return pd.DataFrame()


def main(test_mode: bool = False, n_homes: int = 10):
    print("=" * 60)
    print("Extracting Setpoint Change Responses")
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

        print(f"\nOutput: {OUTPUT_FILE}")
        print(f"  Rows: {len(result):,}")
        print(f"  Episodes: {result['episode_id'].nunique():,}")
        print(f"  By type: {result.groupby('change_type').episode_id.nunique().to_dict()}")
        print(f"  By split: {result.groupby('split').episode_id.nunique().to_dict()}")
    else:
        print("No episodes found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-homes", type=int, default=10)
    args = parser.parse_args()

    main(test_mode=args.test, n_homes=args.n_homes)
