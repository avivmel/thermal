"""
Analyze how often homes are outside setpoint deadband and episode durations.

Questions:
1. What % of time is each home in heating/cooling/passive mode?
2. How long do heating/cooling episodes last?
3. How far outside setpoint do homes get?
"""

import polars as pl
import numpy as np
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = "data/thermal_dataset.csv"


def analyze_mode_episodes(home_df: pl.DataFrame) -> dict:
    """Analyze mode episodes for a single home."""
    home_df = home_df.sort("timestamp")

    indoor = home_df["Indoor_AverageTemperature"].to_numpy()
    heat_sp = home_df["Indoor_HeatSetpoint"].to_numpy()
    cool_sp = home_df["Indoor_CoolSetpoint"].to_numpy()

    n = len(indoor)

    # Determine mode for each timestep
    modes = []
    gaps = []  # Distance from setpoint when in active mode

    for i in range(n):
        t_in = indoor[i]
        t_heat = heat_sp[i]
        t_cool = cool_sp[i]

        if np.isnan(t_in) or t_in == 0 or np.isnan(t_heat) or np.isnan(t_cool):
            modes.append("invalid")
            gaps.append(0)
        elif t_in < t_heat:
            modes.append("heating")
            gaps.append(t_heat - t_in)
        elif t_in > t_cool:
            modes.append("cooling")
            gaps.append(t_in - t_cool)
        else:
            modes.append("passive")
            gaps.append(0)

    modes = np.array(modes)
    gaps = np.array(gaps)

    # Count mode distribution
    valid_mask = modes != "invalid"
    valid_modes = modes[valid_mask]
    n_valid = len(valid_modes)

    if n_valid == 0:
        return None

    mode_counts = {
        "heating": (valid_modes == "heating").sum(),
        "cooling": (valid_modes == "cooling").sum(),
        "passive": (valid_modes == "passive").sum(),
    }

    # Find episodes (contiguous runs of same mode)
    episodes = {"heating": [], "cooling": []}

    current_mode = None
    episode_start = 0
    episode_gaps = []

    for i, mode in enumerate(modes):
        if mode != current_mode:
            # End previous episode
            if current_mode in ["heating", "cooling"]:
                duration = i - episode_start
                max_gap = max(episode_gaps) if episode_gaps else 0
                episodes[current_mode].append({
                    "duration": duration,
                    "max_gap": max_gap,
                    "mean_gap": np.mean(episode_gaps) if episode_gaps else 0,
                })

            # Start new episode
            current_mode = mode
            episode_start = i
            episode_gaps = [gaps[i]] if mode in ["heating", "cooling"] else []
        else:
            if mode in ["heating", "cooling"]:
                episode_gaps.append(gaps[i])

    # Don't forget last episode
    if current_mode in ["heating", "cooling"]:
        duration = len(modes) - episode_start
        max_gap = max(episode_gaps) if episode_gaps else 0
        episodes[current_mode].append({
            "duration": duration,
            "max_gap": max_gap,
            "mean_gap": np.mean(episode_gaps) if episode_gaps else 0,
        })

    return {
        "n_valid": n_valid,
        "mode_counts": mode_counts,
        "episodes": episodes,
        "gaps": gaps[valid_mask],
        "modes": valid_modes,
    }


def main():
    print("Loading data...")
    df = pl.scan_csv(DATA_PATH).collect()
    print(f"Loaded {len(df):,} rows")

    homes = df["home_id"].unique().to_list()
    print(f"Total homes: {len(homes)}")

    # Aggregate stats
    all_heating_episodes = []
    all_cooling_episodes = []
    home_mode_pcts = {"heating": [], "cooling": [], "passive": []}
    all_heating_gaps = []
    all_cooling_gaps = []

    print("\nAnalyzing homes...")
    for home_id in tqdm(homes):
        home_df = df.filter(pl.col("home_id") == home_id)
        result = analyze_mode_episodes(home_df)

        if result is None:
            continue

        n = result["n_valid"]
        for mode in ["heating", "cooling", "passive"]:
            pct = 100 * result["mode_counts"][mode] / n
            home_mode_pcts[mode].append(pct)

        all_heating_episodes.extend(result["episodes"]["heating"])
        all_cooling_episodes.extend(result["episodes"]["cooling"])

        # Collect gaps
        heating_mask = result["modes"] == "heating"
        cooling_mask = result["modes"] == "cooling"
        all_heating_gaps.extend(result["gaps"][heating_mask])
        all_cooling_gaps.extend(result["gaps"][cooling_mask])

    # ==========================================================================
    # Report Results
    # ==========================================================================

    print("\n" + "=" * 70)
    print("MODE DISTRIBUTION ACROSS HOMES")
    print("=" * 70)

    for mode in ["passive", "heating", "cooling"]:
        pcts = home_mode_pcts[mode]
        print(f"\n{mode.upper()} mode (% of time per home):")
        print(f"  Mean:   {np.mean(pcts):5.1f}%")
        print(f"  Median: {np.median(pcts):5.1f}%")
        print(f"  Std:    {np.std(pcts):5.1f}%")
        print(f"  Min:    {np.min(pcts):5.1f}%")
        print(f"  Max:    {np.max(pcts):5.1f}%")
        print(f"  P10:    {np.percentile(pcts, 10):5.1f}%")
        print(f"  P90:    {np.percentile(pcts, 90):5.1f}%")

    # Episode analysis
    print("\n" + "=" * 70)
    print("HEATING EPISODE ANALYSIS")
    print("=" * 70)

    if all_heating_episodes:
        durations = [e["duration"] for e in all_heating_episodes]
        max_gaps = [e["max_gap"] for e in all_heating_episodes]
        mean_gaps = [e["mean_gap"] for e in all_heating_episodes]

        # Convert duration to minutes (5-min intervals)
        durations_min = [d * 5 for d in durations]

        print(f"\nTotal episodes: {len(durations):,}")
        print(f"\nDuration (minutes):")
        print(f"  Mean:   {np.mean(durations_min):6.1f}")
        print(f"  Median: {np.median(durations_min):6.1f}")
        print(f"  Std:    {np.std(durations_min):6.1f}")
        print(f"  P10:    {np.percentile(durations_min, 10):6.1f}")
        print(f"  P25:    {np.percentile(durations_min, 25):6.1f}")
        print(f"  P75:    {np.percentile(durations_min, 75):6.1f}")
        print(f"  P90:    {np.percentile(durations_min, 90):6.1f}")
        print(f"  P99:    {np.percentile(durations_min, 99):6.1f}")
        print(f"  Max:    {np.max(durations_min):6.1f}")

        print(f"\nMax gap from setpoint during episode (°F):")
        print(f"  Mean:   {np.mean(max_gaps):5.2f}")
        print(f"  Median: {np.median(max_gaps):5.2f}")
        print(f"  P90:    {np.percentile(max_gaps, 90):5.2f}")
        print(f"  P99:    {np.percentile(max_gaps, 99):5.2f}")
        print(f"  Max:    {np.max(max_gaps):5.2f}")

        # Duration distribution buckets
        print(f"\nDuration distribution:")
        buckets = [5, 10, 15, 30, 60, 120, 240, float('inf')]
        bucket_names = ["≤5min", "5-10min", "10-15min", "15-30min", "30-60min", "1-2hr", "2-4hr", ">4hr"]
        durations_min = np.array(durations_min)
        prev = 0
        for i, (bucket, name) in enumerate(zip(buckets, bucket_names)):
            if bucket == float('inf'):
                count = (durations_min > prev).sum()
            else:
                count = ((durations_min > prev) & (durations_min <= bucket)).sum()
            pct = 100 * count / len(durations_min)
            print(f"  {name:>10}: {count:>7,} ({pct:5.1f}%)")
            prev = bucket

    print("\n" + "=" * 70)
    print("COOLING EPISODE ANALYSIS")
    print("=" * 70)

    if all_cooling_episodes:
        durations = [e["duration"] for e in all_cooling_episodes]
        max_gaps = [e["max_gap"] for e in all_cooling_episodes]

        durations_min = [d * 5 for d in durations]

        print(f"\nTotal episodes: {len(durations):,}")
        print(f"\nDuration (minutes):")
        print(f"  Mean:   {np.mean(durations_min):6.1f}")
        print(f"  Median: {np.median(durations_min):6.1f}")
        print(f"  Std:    {np.std(durations_min):6.1f}")
        print(f"  P10:    {np.percentile(durations_min, 10):6.1f}")
        print(f"  P25:    {np.percentile(durations_min, 25):6.1f}")
        print(f"  P75:    {np.percentile(durations_min, 75):6.1f}")
        print(f"  P90:    {np.percentile(durations_min, 90):6.1f}")
        print(f"  P99:    {np.percentile(durations_min, 99):6.1f}")
        print(f"  Max:    {np.max(durations_min):6.1f}")

        print(f"\nMax gap from setpoint during episode (°F):")
        print(f"  Mean:   {np.mean(max_gaps):5.2f}")
        print(f"  Median: {np.median(max_gaps):5.2f}")
        print(f"  P90:    {np.percentile(max_gaps, 90):5.2f}")
        print(f"  P99:    {np.percentile(max_gaps, 99):5.2f}")
        print(f"  Max:    {np.max(max_gaps):5.2f}")

        # Duration distribution buckets
        print(f"\nDuration distribution:")
        durations_min = np.array(durations_min)
        prev = 0
        for i, (bucket, name) in enumerate(zip(buckets, bucket_names)):
            if bucket == float('inf'):
                count = (durations_min > prev).sum()
            else:
                count = ((durations_min > prev) & (durations_min <= bucket)).sum()
            pct = 100 * count / len(durations_min)
            print(f"  {name:>10}: {count:>7,} ({pct:5.1f}%)")
            prev = bucket

    # Gap analysis
    print("\n" + "=" * 70)
    print("GAP FROM SETPOINT ANALYSIS (when outside deadband)")
    print("=" * 70)

    if all_heating_gaps:
        gaps = np.array(all_heating_gaps)
        print(f"\nHEATING (indoor < heat_setpoint):")
        print(f"  Total samples: {len(gaps):,}")
        print(f"  Gap = heat_sp - indoor")
        print(f"  Mean:   {np.mean(gaps):5.2f}°F")
        print(f"  Median: {np.median(gaps):5.2f}°F")
        print(f"  Std:    {np.std(gaps):5.2f}°F")
        print(f"  P10:    {np.percentile(gaps, 10):5.2f}°F")
        print(f"  P90:    {np.percentile(gaps, 90):5.2f}°F")
        print(f"  P99:    {np.percentile(gaps, 99):5.2f}°F")
        print(f"  Max:    {np.max(gaps):5.2f}°F")

        # Gap buckets
        print(f"\n  Gap distribution:")
        gap_buckets = [0.5, 1, 2, 3, 5, 10, float('inf')]
        gap_names = ["≤0.5°F", "0.5-1°F", "1-2°F", "2-3°F", "3-5°F", "5-10°F", ">10°F"]
        prev = 0
        for bucket, name in zip(gap_buckets, gap_names):
            if bucket == float('inf'):
                count = (gaps > prev).sum()
            else:
                count = ((gaps > prev) & (gaps <= bucket)).sum()
            pct = 100 * count / len(gaps)
            print(f"    {name:>10}: {count:>10,} ({pct:5.1f}%)")
            prev = bucket

    if all_cooling_gaps:
        gaps = np.array(all_cooling_gaps)
        print(f"\nCOOLING (indoor > cool_setpoint):")
        print(f"  Total samples: {len(gaps):,}")
        print(f"  Gap = indoor - cool_sp")
        print(f"  Mean:   {np.mean(gaps):5.2f}°F")
        print(f"  Median: {np.median(gaps):5.2f}°F")
        print(f"  Std:    {np.std(gaps):5.2f}°F")
        print(f"  P10:    {np.percentile(gaps, 10):5.2f}°F")
        print(f"  P90:    {np.percentile(gaps, 90):5.2f}°F")
        print(f"  P99:    {np.percentile(gaps, 99):5.2f}°F")
        print(f"  Max:    {np.max(gaps):5.2f}°F")

        print(f"\n  Gap distribution:")
        prev = 0
        for bucket, name in zip(gap_buckets, gap_names):
            if bucket == float('inf'):
                count = (gaps > prev).sum()
            else:
                count = ((gaps > prev) & (gaps <= bucket)).sum()
            pct = 100 * count / len(gaps)
            print(f"    {name:>10}: {count:>10,} ({pct:5.1f}%)")
            prev = bucket

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    heating_pcts = home_mode_pcts["heating"]
    cooling_pcts = home_mode_pcts["cooling"]
    passive_pcts = home_mode_pcts["passive"]

    print(f"\n1. Mode time varies hugely across homes:")
    print(f"   - Heating: {np.min(heating_pcts):.0f}% to {np.max(heating_pcts):.0f}% (median {np.median(heating_pcts):.0f}%)")
    print(f"   - Cooling: {np.min(cooling_pcts):.0f}% to {np.max(cooling_pcts):.0f}% (median {np.median(cooling_pcts):.0f}%)")

    if all_heating_episodes:
        heating_durations = [e["duration"] * 5 for e in all_heating_episodes]
        short_heating = sum(1 for d in heating_durations if d <= 30)
        print(f"\n2. Heating episodes:")
        print(f"   - Median duration: {np.median(heating_durations):.0f} min")
        print(f"   - {100*short_heating/len(heating_durations):.0f}% are ≤30 min")

    if all_cooling_episodes:
        cooling_durations = [e["duration"] * 5 for e in all_cooling_episodes]
        short_cooling = sum(1 for d in cooling_durations if d <= 30)
        print(f"\n3. Cooling episodes:")
        print(f"   - Median duration: {np.median(cooling_durations):.0f} min")
        print(f"   - {100*short_cooling/len(cooling_durations):.0f}% are ≤30 min")

    if all_heating_gaps:
        small_heating_gap = sum(1 for g in all_heating_gaps if g <= 1)
        print(f"\n4. Heating gaps are usually small:")
        print(f"   - Median gap: {np.median(all_heating_gaps):.2f}°F")
        print(f"   - {100*small_heating_gap/len(all_heating_gaps):.0f}% are ≤1°F from setpoint")


if __name__ == "__main__":
    main()
