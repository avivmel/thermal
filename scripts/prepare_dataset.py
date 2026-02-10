"""
Prepare thermal prediction dataset from Ecobee NetCDF files.

Creates train/val/test splits (70/10/20) stratified by state.
Outputs a single CSV with all data.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# Paths
DATA_DIR = Path("ecobee_processed_dataset/extracted/clean_data")
OUTPUT_PATH = Path("data/thermal_dataset.csv")

# Columns to extract
COLUMNS = [
    "Indoor_AverageTemperature",
    "Outdoor_Temperature",
    "Indoor_HeatSetpoint",
    "Indoor_CoolSetpoint",
]

# Month files in order
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


def get_home_states(ds: xr.Dataset) -> dict:
    """Extract state for each home (constant per home)."""
    home_ids = ds.id.values
    states = ds["State"].isel(time=0).values
    return dict(zip(home_ids, states))


def create_splits(home_ids: np.ndarray, states: np.ndarray, seed: int = 42) -> dict:
    """
    Create train/val/test splits stratified by state.

    Split: 70% train, 10% val, 20% test
    """
    # First split: 80% train+val, 20% test
    train_val_ids, test_ids = train_test_split(
        home_ids,
        test_size=0.20,
        stratify=states,
        random_state=seed
    )

    # Get states for train_val
    train_val_states = states[np.isin(home_ids, train_val_ids)]

    # Second split: 70/10 from the 80% = 87.5% train, 12.5% val
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.125,  # 10% of total = 12.5% of 80%
        stratify=train_val_states,
        random_state=seed
    )

    # Create mapping
    split_map = {}
    for hid in train_ids:
        split_map[hid] = "train"
    for hid in val_ids:
        split_map[hid] = "val"
    for hid in test_ids:
        split_map[hid] = "test"

    return split_map


def process_month_fast(month: str, home_states: dict, split_map: dict) -> pd.DataFrame:
    """Process a single month's data (vectorized)."""
    filepath = DATA_DIR / f"{month}_clean.nc"

    ds = xr.open_dataset(filepath)

    # Filter to homes with valid states
    valid_homes = [hid for hid in ds.id.values if hid in split_map]
    ds = ds.sel(id=valid_homes)

    home_ids = ds.id.values
    times = ds.time.values
    n_homes = len(home_ids)
    n_times = len(times)

    # Create index arrays
    home_idx = np.repeat(np.arange(n_homes), n_times)
    time_idx = np.tile(np.arange(n_times), n_homes)

    # Build dataframe
    df = pd.DataFrame({
        "home_id": home_ids[home_idx],
        "timestamp": times[time_idx],
    })

    # Add state and split
    df["state"] = df["home_id"].map(home_states)
    df["split"] = df["home_id"].map(split_map)

    # Add data columns
    for col in COLUMNS:
        data = ds[col].values  # (n_homes, n_times)
        df[col] = data.flatten()

    ds.close()
    return df


def main():
    print("=" * 60)
    print("Preparing Thermal Prediction Dataset")
    print("=" * 60)

    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load first month to get home info
    print("\n[1/4] Loading home metadata...")
    ds = xr.open_dataset(DATA_DIR / "Jan_clean.nc")
    home_states = get_home_states(ds)
    ds.close()

    # Filter homes with valid states
    valid_homes = {hid: state for hid, state in home_states.items() if state != ""}
    print(f"  Valid homes: {len(valid_homes)} / {len(home_states)}")

    home_ids = np.array(list(valid_homes.keys()))
    states = np.array(list(valid_homes.values()))

    # Create splits
    print("\n[2/4] Creating train/val/test splits...")
    split_map = create_splits(home_ids, states)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for split in split_map.values():
        split_counts[split] += 1

    print(f"  Train: {split_counts['train']} | Val: {split_counts['val']} | Test: {split_counts['test']}")

    # Process each month and write incrementally
    print("\n[3/4] Processing months...")
    total_rows = 0
    first_write = True

    for month in tqdm(MONTHS, desc="Months"):
        df = process_month_fast(month, valid_homes, split_map)

        # Write to CSV (append mode after first write)
        df.to_csv(
            OUTPUT_PATH,
            mode='w' if first_write else 'a',
            header=first_write,
            index=False
        )
        first_write = False
        total_rows += len(df)

    # Summary
    print(f"\n[4/4] Done!")
    file_size = OUTPUT_PATH.stat().st_size / (1024 ** 3)
    print(f"\n{'=' * 60}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {file_size:.2f} GB")
    print(f"Rows: {total_rows:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
