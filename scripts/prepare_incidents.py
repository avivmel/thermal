"""
Extract goal-oriented thermal incidents from raw Ecobee NetCDF data.

Creates parquet files with incident metadata for:
- Heating episodes: indoor_temp drops below heat_setpoint
- Cooling episodes: indoor_temp rises above cool_setpoint
- Drift episodes: HVAC goal achieved, temp drifting in deadband
- Setpoint change episodes: user/schedule changes setpoint

Output format: One row per timestep within an incident, with added incident metadata.

Note: Episodes that span month boundaries are split into separate episodes.
This is acceptable since most episodes are short (median ~25 min for heating).

Usage:
    python scripts/prepare_incidents.py           # Full run (all months, all homes)
    python scripts/prepare_incidents.py --test    # Test mode (Jan only, 5 homes)
    python scripts/prepare_incidents.py --test --n-homes 10  # Test with 10 homes
"""

import argparse
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
DATA_DIR = Path("ecobee_processed_dataset/extracted/clean_data")
OUTPUT_DIR = Path("data/incidents")
THERMAL_DATASET = Path("data/thermal_dataset.csv")

# Month files in order
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# All columns to preserve from NetCDF
PRESERVED_COLUMNS = [
    # Temperatures
    "Indoor_AverageTemperature",
    "Outdoor_Temperature",
    "Thermostat_Temperature",
    "RemoteSensor1_Temperature",
    "RemoteSensor2_Temperature",
    "RemoteSensor3_Temperature",
    "RemoteSensor4_Temperature",
    "RemoteSensor5_Temperature",
    # Setpoints
    "Indoor_HeatSetpoint",
    "Indoor_CoolSetpoint",
    # Humidity
    "Indoor_Humidity",
    "Outdoor_Humidity",
    # Runtime
    "HeatingEquipmentStage1_RunTime",
    "HeatingEquipmentStage2_RunTime",
    "HeatingEquipmentStage3_RunTime",
    "CoolingEquipmentStage1_RunTime",
    "CoolingEquipmentStage2_RunTime",
    "HeatPumpsStage1_RunTime",
    "HeatPumpsStage2_RunTime",
    "Fan_RunTime",
    # Motion
    "Thermostat_DetectedMotion",
    "RemoteSensor1_DetectedMotion",
    "RemoteSensor2_DetectedMotion",
    "RemoteSensor3_DetectedMotion",
    "RemoteSensor4_DetectedMotion",
    "RemoteSensor5_DetectedMotion",
    # Other
    "Schedule",
    "Event",
    "HVAC_Mode",
]

# Episode timeout limits (in timesteps, each 5 minutes)
DRIFT_TIMEOUT = 48      # 4 hours
SETPOINT_TIMEOUT = 24   # 2 hours

# Setpoint change threshold (°F)
SETPOINT_THRESHOLD = 0.5


@dataclass
class Episode:
    """Represents a thermal episode in progress."""
    incident_type: str
    start_idx: int
    start_timestamp: pd.Timestamp
    start_indoor_temp: float
    start_outdoor_temp: float
    target_setpoint: float
    initial_gap: float
    timestep_indices: list


def classify_mode(indoor: float, heat_sp: float, cool_sp: float) -> str:
    """Classify thermal mode based on setpoint comparison."""
    if indoor < heat_sp:
        return "heating"
    elif indoor > cool_sp:
        return "cooling"
    else:
        return "deadband"


def is_active(incident_type: str, indoor: float, outdoor: float) -> bool:
    """Determine if HVAC is fighting thermal gradient."""
    if incident_type == "heating":
        return outdoor < indoor  # HVAC fighting cold outside
    elif incident_type == "cooling":
        return outdoor > indoor  # HVAC fighting hot outside
    return False


def detect_setpoint_change(
    heat_sp: float,
    prev_heat_sp: float,
    cool_sp: float,
    prev_cool_sp: float
) -> tuple[bool, str, float]:
    """
    Detect meaningful setpoint changes.

    Returns: (changed, direction, new_target)
        direction: 'heat_increase' or 'cool_decrease'
    """
    heat_increased = heat_sp > prev_heat_sp + SETPOINT_THRESHOLD
    cool_decreased = cool_sp < prev_cool_sp - SETPOINT_THRESHOLD

    if heat_increased:
        return True, "heat_increase", heat_sp
    elif cool_decreased:
        return True, "cool_decrease", cool_sp
    return False, "", 0.0


def load_home_splits() -> dict:
    """Load home-to-split mapping from existing thermal dataset."""
    print("Loading home splits from thermal_dataset.csv...")
    # Read in chunks to extract unique home_id -> split mapping without loading all 101M rows
    splits = {}
    chunk_size = 1_000_000

    for chunk in pd.read_csv(
        THERMAL_DATASET,
        usecols=["home_id", "split"],
        chunksize=chunk_size
    ):
        # Get unique mappings from this chunk
        chunk_splits = chunk.drop_duplicates("home_id").set_index("home_id")["split"].to_dict()
        # Add new homes (don't overwrite - first occurrence wins)
        for home_id, split in chunk_splits.items():
            if home_id not in splits:
                splits[home_id] = split

        # Early exit if we've likely found all homes (971 expected)
        if len(splits) >= 971:
            break

    print(f"  Loaded splits for {len(splits)} homes")
    return splits


def load_home_states(ds: xr.Dataset) -> dict:
    """Extract state for each home (constant per home)."""
    home_ids = ds.id.values
    states = ds["State"].isel(time=0).values
    return dict(zip(home_ids, states))


def process_home(
    home_id: str,
    home_data: dict,
    times: np.ndarray,
    state: str,
    split: str
) -> dict[str, list]:
    """
    Process a single home's data to extract incidents.

    Args:
        home_id: Home identifier
        home_data: Dictionary of column name -> numpy array for this home
        times: Array of timestamps
        state: Home's state code
        split: train/val/test

    Returns:
        Dictionary with keys 'heating', 'cooling', 'drift', 'setpoint'
        containing lists of row dictionaries
    """
    incidents = {
        "heating": [],
        "cooling": [],
        "drift": [],
        "setpoint": []
    }

    indoor = home_data["Indoor_AverageTemperature"]
    outdoor = home_data["Outdoor_Temperature"]
    heat_sp = home_data["Indoor_HeatSetpoint"]
    cool_sp = home_data["Indoor_CoolSetpoint"]

    n_times = len(times)
    current_episode: Optional[Episode] = None
    prev_mode = None

    def close_episode(end_idx: int, end_timestamp: pd.Timestamp):
        """Close current episode and add rows to incidents."""
        nonlocal current_episode
        if current_episode is None:
            return

        episode = current_episode
        incident_id = f"{home_id}_{episode.incident_type}_{episode.start_timestamp.isoformat()}"
        episode_duration = len(episode.timestep_indices)

        # Add all timesteps in this episode
        for timestep_idx, t_idx in enumerate(episode.timestep_indices):
            row = {
                # Incident metadata
                "incident_id": incident_id,
                "home_id": home_id,
                "timestamp": pd.Timestamp(times[t_idx]),
                "state": state,
                "split": split,
                "incident_type": episode.incident_type,
                "is_active": is_active(
                    episode.incident_type,
                    indoor[t_idx],
                    outdoor[t_idx]
                ),
                "timestep_idx": timestep_idx,
                "episode_duration": episode_duration,
                "episode_start": episode.start_timestamp,
                "episode_end": end_timestamp,
                "start_indoor_temp": episode.start_indoor_temp,
                "start_outdoor_temp": episode.start_outdoor_temp,
                "target_setpoint": episode.target_setpoint,
                "initial_gap": episode.initial_gap,
            }

            # Add all preserved columns
            for col in PRESERVED_COLUMNS:
                if col in home_data:
                    row[col] = home_data[col][t_idx]

            # Determine output category
            if episode.incident_type == "setpoint_change":
                incidents["setpoint"].append(row)
            else:
                incidents[episode.incident_type].append(row)

        current_episode = None

    def start_episode(
        incident_type: str,
        t_idx: int,
        target_sp: float
    ):
        """Start a new episode."""
        nonlocal current_episode

        indoor_temp = indoor[t_idx]
        outdoor_temp = outdoor[t_idx]

        # Calculate initial gap
        if incident_type in ("heating", "setpoint_change") and heat_sp[t_idx] > 0:
            gap = target_sp - indoor_temp
        elif incident_type == "cooling" and cool_sp[t_idx] > 0:
            gap = indoor_temp - target_sp
        else:
            gap = 0.0

        current_episode = Episode(
            incident_type=incident_type,
            start_idx=t_idx,
            start_timestamp=pd.Timestamp(times[t_idx]),
            start_indoor_temp=indoor_temp,
            start_outdoor_temp=outdoor_temp,
            target_setpoint=target_sp,
            initial_gap=abs(gap),
            timestep_indices=[t_idx]
        )

    for t_idx in range(n_times):
        # Skip if missing data (0.0 indicates missing for temps)
        if indoor[t_idx] == 0.0 or heat_sp[t_idx] == 0.0 or cool_sp[t_idx] == 0.0:
            # Close any open episode at data gap
            if current_episode is not None:
                close_episode(t_idx - 1, pd.Timestamp(times[t_idx - 1]))
            prev_mode = None
            continue

        mode = classify_mode(indoor[t_idx], heat_sp[t_idx], cool_sp[t_idx])

        # Check for setpoint change (only if not first timestep)
        setpoint_changed = False
        change_direction = ""
        new_target = 0.0

        if t_idx > 0 and heat_sp[t_idx - 1] > 0 and cool_sp[t_idx - 1] > 0:
            setpoint_changed, change_direction, new_target = detect_setpoint_change(
                heat_sp[t_idx], heat_sp[t_idx - 1],
                cool_sp[t_idx], cool_sp[t_idx - 1]
            )

        # Handle episode transitions
        if setpoint_changed:
            # Close current episode
            close_episode(t_idx - 1, pd.Timestamp(times[t_idx - 1]))
            # Start setpoint change episode
            start_episode("setpoint_change", t_idx, new_target)

        elif mode != prev_mode:
            # Mode transition
            close_episode(t_idx - 1 if t_idx > 0 else 0,
                         pd.Timestamp(times[t_idx - 1] if t_idx > 0 else times[0]))

            if mode == "heating":
                start_episode("heating", t_idx, heat_sp[t_idx])
            elif mode == "cooling":
                start_episode("cooling", t_idx, cool_sp[t_idx])
            elif mode == "deadband" and prev_mode in ("heating", "cooling"):
                # Starting drift from an HVAC episode
                start_episode("drift", t_idx, 0.0)

        # Add timestep to current episode
        if current_episode is not None:
            if t_idx != current_episode.start_idx:
                current_episode.timestep_indices.append(t_idx)

            # Check for episode timeout
            episode_len = len(current_episode.timestep_indices)
            if current_episode.incident_type == "drift" and episode_len >= DRIFT_TIMEOUT:
                close_episode(t_idx, pd.Timestamp(times[t_idx]))
            elif current_episode.incident_type == "setpoint_change":
                # Check if target reached or timeout
                reached_target = False
                if change_direction == "heat_increase":
                    reached_target = indoor[t_idx] >= current_episode.target_setpoint
                elif change_direction == "cool_decrease":
                    reached_target = indoor[t_idx] <= current_episode.target_setpoint

                if reached_target or episode_len >= SETPOINT_TIMEOUT:
                    close_episode(t_idx, pd.Timestamp(times[t_idx]))

        prev_mode = mode

    # Close any remaining episode at end of data
    if current_episode is not None:
        close_episode(n_times - 1, pd.Timestamp(times[n_times - 1]))

    return incidents


def process_month(
    month: str,
    home_splits: dict,
    output_dir: Path,
    max_homes: Optional[int] = None
) -> dict[str, int]:
    """
    Process a single month's data.

    Args:
        month: Month name (e.g., "Jan")
        home_splits: Home ID -> split mapping
        output_dir: Directory to write parquet files
        max_homes: Optional limit on number of homes to process (for testing)

    Returns:
        Dictionary of row counts by incident type
    """
    filepath = DATA_DIR / f"{month}_clean.nc"
    ds = xr.open_dataset(filepath)

    # Get home states
    home_states = load_home_states(ds)

    # Filter to homes with valid splits
    valid_homes = [hid for hid in ds.id.values if hid in home_splits]
    if max_homes is not None:
        valid_homes = valid_homes[:max_homes]
    ds = ds.sel(id=valid_homes)

    home_ids = ds.id.values
    times = pd.to_datetime(ds.time.values)

    # Pre-load all data for efficiency
    all_data = {}
    for col in PRESERVED_COLUMNS:
        if col in ds:
            all_data[col] = ds[col].values  # (n_homes, n_times)

    # Accumulate incidents for this month
    month_incidents = {"heating": [], "cooling": [], "drift": [], "setpoint": []}
    counts = {"heating": 0, "cooling": 0, "drift": 0, "setpoint": 0}

    for home_idx, home_id in enumerate(tqdm(home_ids, desc=f"  {month}", leave=False)):
        # Extract this home's data
        home_data = {}
        for col, data in all_data.items():
            home_data[col] = data[home_idx]

        state = home_states.get(home_id, "")
        split = home_splits.get(home_id, "")

        if not state or not split:
            continue

        # Process home
        incidents = process_home(home_id, home_data, times.values, state, split)

        # Accumulate incidents
        for incident_type, rows in incidents.items():
            if rows:
                month_incidents[incident_type].extend(rows)
                counts[incident_type] += len(rows)

    ds.close()

    # Write accumulated incidents to per-month temporary files
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(exist_ok=True)

    for incident_type, rows in month_incidents.items():
        if rows:
            df = pd.DataFrame(rows)
            filepath = temp_dir / f"{incident_type}_{month}.parquet"
            df.to_parquet(filepath, index=False)

    return counts


def merge_temp_files(output_dir: Path):
    """Merge temporary per-month files into final output files."""
    print("\nMerging temporary files...")

    temp_dir = output_dir / "_temp"
    if not temp_dir.exists():
        return

    for incident_type in ["heating", "cooling", "drift", "setpoint"]:
        temp_files = sorted(temp_dir.glob(f"{incident_type}_*.parquet"))
        if not temp_files:
            continue

        # Read and concatenate all temp files
        dfs = [pd.read_parquet(f) for f in temp_files]
        merged = pd.concat(dfs, ignore_index=True)

        # Write final file
        output_file = output_dir / f"{incident_type}.parquet"
        merged.to_parquet(output_file, index=False)
        print(f"  {incident_type}: {len(merged):,} rows from {len(temp_files)} files")

    # Clean up temp directory
    shutil.rmtree(temp_dir)
    print("  Cleaned up temporary files")


def create_summary(output_dir: Path) -> pd.DataFrame:
    """Create summary parquet with one row per episode."""
    print("\nCreating summary file...")

    summary_rows = []

    for incident_type in ["heating", "cooling", "drift", "setpoint"]:
        filepath = output_dir / f"{incident_type}.parquet"
        if not filepath.exists():
            continue

        df = pd.read_parquet(filepath)

        # Group by incident_id and take first row for metadata
        for incident_id, group in df.groupby("incident_id"):
            first = group.iloc[0]
            last = group.iloc[-1]

            summary_rows.append({
                "incident_id": incident_id,
                "home_id": first["home_id"],
                "state": first["state"],
                "split": first["split"],
                "incident_type": first["incident_type"],
                "episode_start": first["episode_start"],
                "episode_end": first["episode_end"],
                "episode_duration": first["episode_duration"],
                "start_indoor_temp": first["start_indoor_temp"],
                "start_outdoor_temp": first["start_outdoor_temp"],
                "end_indoor_temp": last["Indoor_AverageTemperature"],
                "end_outdoor_temp": last["Outdoor_Temperature"],
                "target_setpoint": first["target_setpoint"],
                "initial_gap": first["initial_gap"],
                "pct_active": group["is_active"].mean(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(output_dir / "summary.parquet", index=False)
    print(f"  Summary: {len(summary_df):,} episodes")

    return summary_df


def verify_output(output_dir: Path, home_splits: dict):
    """Verify output integrity."""
    print("\nVerifying output...")

    total_rows = 0
    for incident_type in ["heating", "cooling", "drift", "setpoint"]:
        filepath = output_dir / f"{incident_type}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            total_rows += len(df)

            # Check split preservation
            split_counts = df.groupby("split").size()
            print(f"  {incident_type}: {len(df):,} rows | {split_counts.to_dict()}")

            # Verify splits match original
            for home_id in df["home_id"].unique()[:10]:
                expected = home_splits.get(home_id)
                actual = df[df["home_id"] == home_id]["split"].iloc[0]
                assert expected == actual, f"Split mismatch for {home_id}"

    print(f"\n  Total rows: {total_rows:,}")
    print("  Split verification: PASSED")


def main(test_mode: bool = False, n_homes: int = 5):
    print("=" * 60)
    print("Extracting Thermal Incidents")
    if test_mode:
        print(f"  TEST MODE: Processing {n_homes} homes from January only")
    print("=" * 60)

    # Use test output directory in test mode
    output_dir = OUTPUT_DIR if not test_mode else Path("data/incidents_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up existing files to start fresh
    for incident_type in ["heating", "cooling", "drift", "setpoint"]:
        filepath = output_dir / f"{incident_type}.parquet"
        if filepath.exists():
            filepath.unlink()

    # Load home splits
    home_splits = load_home_splits()

    # Process months
    months_to_process = ["Jan"] if test_mode else MONTHS

    print("\nProcessing months...")
    total_counts = {"heating": 0, "cooling": 0, "drift": 0, "setpoint": 0}

    for month in tqdm(months_to_process, desc="Months"):
        counts = process_month(
            month, home_splits, output_dir,
            max_homes=n_homes if test_mode else None
        )
        for k, v in counts.items():
            total_counts[k] += v

    # Merge temp files into final output
    merge_temp_files(output_dir)

    # Print counts
    print("\nRow counts by incident type:")
    for incident_type, count in total_counts.items():
        filepath = output_dir / f"{incident_type}.parquet"
        size_mb = filepath.stat().st_size / (1024 ** 2) if filepath.exists() else 0
        print(f"  {incident_type}: {count:,} rows ({size_mb:.1f} MB)")

    # Create summary file
    create_summary(output_dir)

    # Verify output
    verify_output(output_dir, home_splits)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract thermal incidents from Ecobee data")
    parser.add_argument("--test", action="store_true", help="Run in test mode (5 homes, Jan only)")
    parser.add_argument("--n-homes", type=int, default=5, help="Number of homes in test mode")
    args = parser.parse_args()

    main(test_mode=args.test, n_homes=args.n_homes)
