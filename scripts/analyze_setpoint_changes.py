"""
Analyze setpoint changes and thermal dynamics.

Questions:
1. How many significant setpoint changes per home?
2. Distribution of homes by change frequency
3. Active vs passive thermal modification (considering outdoor temp)
"""

import polars as pl
import numpy as np
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = "data/thermal_dataset.csv"

# What counts as a "significant" setpoint change
MIN_SETPOINT_CHANGE = 1.0  # °F


def analyze_home(home_df: pl.DataFrame) -> dict:
    """Analyze setpoint changes and thermal dynamics for one home."""
    home_df = home_df.sort("timestamp")

    indoor = home_df["Indoor_AverageTemperature"].to_numpy()
    outdoor = home_df["Outdoor_Temperature"].to_numpy()
    heat_sp = home_df["Indoor_HeatSetpoint"].to_numpy()
    cool_sp = home_df["Indoor_CoolSetpoint"].to_numpy()

    n = len(indoor)
    if n < 100:
        return None

    # Find setpoint changes
    heat_changes = []
    cool_changes = []

    for i in range(1, n):
        if np.isnan(heat_sp[i]) or np.isnan(heat_sp[i-1]):
            continue
        if np.isnan(cool_sp[i]) or np.isnan(cool_sp[i-1]):
            continue

        heat_delta = heat_sp[i] - heat_sp[i-1]
        cool_delta = cool_sp[i] - cool_sp[i-1]

        if abs(heat_delta) >= MIN_SETPOINT_CHANGE:
            heat_changes.append({
                "idx": i,
                "delta": heat_delta,
                "old": heat_sp[i-1],
                "new": heat_sp[i],
                "indoor": indoor[i],
                "outdoor": outdoor[i],
            })

        if abs(cool_delta) >= MIN_SETPOINT_CHANGE:
            cool_changes.append({
                "idx": i,
                "delta": cool_delta,
                "old": cool_sp[i-1],
                "new": cool_sp[i],
                "indoor": indoor[i],
                "outdoor": outdoor[i],
            })

    # Analyze thermal dynamics - active vs passive
    # Active heating: indoor < heat_sp AND outdoor < indoor (fighting thermal gradient)
    # Passive heating: indoor < heat_sp AND outdoor > indoor (thermal gradient helps)
    # Active cooling: indoor > cool_sp AND outdoor > indoor (fighting thermal gradient)
    # Passive cooling: indoor > cool_sp AND outdoor < indoor (thermal gradient helps)

    active_heating = 0
    passive_heating = 0
    active_cooling = 0
    passive_cooling = 0
    passive_mode = 0
    invalid = 0

    for i in range(n):
        t_in = indoor[i]
        t_out = outdoor[i]
        t_heat = heat_sp[i]
        t_cool = cool_sp[i]

        if np.isnan(t_in) or t_in == 0 or np.isnan(t_out) or t_out == 0:
            invalid += 1
            continue
        if np.isnan(t_heat) or np.isnan(t_cool):
            invalid += 1
            continue

        if t_in < t_heat:
            # Heating mode
            if t_out < t_in:
                active_heating += 1  # HVAC fighting cold outside
            else:
                passive_heating += 1  # Warm outside helps
        elif t_in > t_cool:
            # Cooling mode
            if t_out > t_in:
                active_cooling += 1  # HVAC fighting hot outside
            else:
                passive_cooling += 1  # Cool outside helps
        else:
            passive_mode += 1

    total_valid = n - invalid

    return {
        "n_samples": n,
        "n_valid": total_valid,
        "heat_changes": heat_changes,
        "cool_changes": cool_changes,
        "n_heat_changes": len(heat_changes),
        "n_cool_changes": len(cool_changes),
        "active_heating": active_heating,
        "passive_heating": passive_heating,
        "active_cooling": active_cooling,
        "passive_cooling": passive_cooling,
        "passive_mode": passive_mode,
    }


def main():
    print("Loading data...")
    df = pl.scan_csv(DATA_PATH).collect()
    print(f"Loaded {len(df):,} rows")

    homes = df["home_id"].unique().to_list()
    print(f"Total homes: {len(homes)}")
    print(f"Min setpoint change threshold: {MIN_SETPOINT_CHANGE}°F")

    results = []

    print("\nAnalyzing homes...")
    for home_id in tqdm(homes):
        home_df = df.filter(pl.col("home_id") == home_id)
        result = analyze_home(home_df)
        if result:
            result["home_id"] = home_id
            results.append(result)

    print(f"\nAnalyzed {len(results)} homes")

    # ==========================================================================
    # Setpoint Change Analysis
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SETPOINT CHANGE ANALYSIS")
    print("=" * 70)

    # Heat setpoint changes
    heat_change_counts = [r["n_heat_changes"] for r in results]
    cool_change_counts = [r["n_cool_changes"] for r in results]
    total_change_counts = [r["n_heat_changes"] + r["n_cool_changes"] for r in results]

    print(f"\n### HEAT SETPOINT CHANGES (≥{MIN_SETPOINT_CHANGE}°F)")
    print(f"Total changes across all homes: {sum(heat_change_counts):,}")
    print(f"\nChanges per home:")
    print(f"  Mean:   {np.mean(heat_change_counts):,.1f}")
    print(f"  Median: {np.median(heat_change_counts):,.0f}")
    print(f"  Std:    {np.std(heat_change_counts):,.1f}")
    print(f"  Min:    {np.min(heat_change_counts):,}")
    print(f"  Max:    {np.max(heat_change_counts):,}")
    print(f"  P10:    {np.percentile(heat_change_counts, 10):,.0f}")
    print(f"  P90:    {np.percentile(heat_change_counts, 90):,.0f}")

    # Homes with 0 changes
    zero_heat = sum(1 for c in heat_change_counts if c == 0)
    print(f"\nHomes with 0 heat setpoint changes: {zero_heat} ({100*zero_heat/len(results):.1f}%)")

    print(f"\n### COOL SETPOINT CHANGES (≥{MIN_SETPOINT_CHANGE}°F)")
    print(f"Total changes across all homes: {sum(cool_change_counts):,}")
    print(f"\nChanges per home:")
    print(f"  Mean:   {np.mean(cool_change_counts):,.1f}")
    print(f"  Median: {np.median(cool_change_counts):,.0f}")
    print(f"  Std:    {np.std(cool_change_counts):,.1f}")
    print(f"  Min:    {np.min(cool_change_counts):,}")
    print(f"  Max:    {np.max(cool_change_counts):,}")
    print(f"  P10:    {np.percentile(cool_change_counts, 10):,.0f}")
    print(f"  P90:    {np.percentile(cool_change_counts, 90):,.0f}")

    zero_cool = sum(1 for c in cool_change_counts if c == 0)
    print(f"\nHomes with 0 cool setpoint changes: {zero_cool} ({100*zero_cool/len(results):.1f}%)")

    print(f"\n### TOTAL SETPOINT CHANGES (heat + cool)")
    print(f"Total changes across all homes: {sum(total_change_counts):,}")
    print(f"\nChanges per home:")
    print(f"  Mean:   {np.mean(total_change_counts):,.1f}")
    print(f"  Median: {np.median(total_change_counts):,.0f}")
    print(f"  P10:    {np.percentile(total_change_counts, 10):,.0f}")
    print(f"  P90:    {np.percentile(total_change_counts, 90):,.0f}")

    # Distribution buckets
    print(f"\n### HOMES BY SETPOINT CHANGE FREQUENCY")
    buckets = [(0, 0), (1, 10), (11, 50), (51, 100), (101, 500), (501, 1000), (1001, float('inf'))]
    bucket_names = ["0", "1-10", "11-50", "51-100", "101-500", "501-1000", ">1000"]

    print(f"\nTotal changes per home (heat + cool):")
    for (lo, hi), name in zip(buckets, bucket_names):
        if hi == float('inf'):
            count = sum(1 for c in total_change_counts if c > lo)
        else:
            count = sum(1 for c in total_change_counts if lo <= c <= hi)
        pct = 100 * count / len(results)
        print(f"  {name:>10}: {count:>4} homes ({pct:5.1f}%)")

    # ==========================================================================
    # Change magnitude analysis
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SETPOINT CHANGE MAGNITUDE")
    print("=" * 70)

    all_heat_deltas = []
    all_cool_deltas = []
    for r in results:
        all_heat_deltas.extend([c["delta"] for c in r["heat_changes"]])
        all_cool_deltas.extend([c["delta"] for c in r["cool_changes"]])

    if all_heat_deltas:
        deltas = np.array(all_heat_deltas)
        print(f"\nHeat setpoint change magnitudes ({len(deltas):,} changes):")
        print(f"  Mean:   {np.mean(deltas):+.2f}°F")
        print(f"  Median: {np.median(deltas):+.2f}°F")
        print(f"  Std:    {np.std(deltas):.2f}°F")
        print(f"  Min:    {np.min(deltas):+.2f}°F")
        print(f"  Max:    {np.max(deltas):+.2f}°F")

        # Up vs down
        up = (deltas > 0).sum()
        down = (deltas < 0).sum()
        print(f"\n  Increases (warmer): {up:,} ({100*up/len(deltas):.1f}%)")
        print(f"  Decreases (cooler): {down:,} ({100*down/len(deltas):.1f}%)")

        # Magnitude buckets
        abs_deltas = np.abs(deltas)
        print(f"\n  By magnitude:")
        mag_buckets = [(1, 2), (2, 3), (3, 5), (5, 10), (10, float('inf'))]
        mag_names = ["1-2°F", "2-3°F", "3-5°F", "5-10°F", ">10°F"]
        for (lo, hi), name in zip(mag_buckets, mag_names):
            if hi == float('inf'):
                count = ((abs_deltas >= lo)).sum()
            else:
                count = ((abs_deltas >= lo) & (abs_deltas < hi)).sum()
            pct = 100 * count / len(abs_deltas)
            print(f"    {name:>8}: {count:>7,} ({pct:5.1f}%)")

    if all_cool_deltas:
        deltas = np.array(all_cool_deltas)
        print(f"\nCool setpoint change magnitudes ({len(deltas):,} changes):")
        print(f"  Mean:   {np.mean(deltas):+.2f}°F")
        print(f"  Median: {np.median(deltas):+.2f}°F")
        print(f"  Std:    {np.std(deltas):.2f}°F")
        print(f"  Min:    {np.min(deltas):+.2f}°F")
        print(f"  Max:    {np.max(deltas):+.2f}°F")

        up = (deltas > 0).sum()
        down = (deltas < 0).sum()
        print(f"\n  Increases (warmer): {up:,} ({100*up/len(deltas):.1f}%)")
        print(f"  Decreases (cooler): {down:,} ({100*down/len(deltas):.1f}%)")

        abs_deltas = np.abs(deltas)
        print(f"\n  By magnitude:")
        for (lo, hi), name in zip(mag_buckets, mag_names):
            if hi == float('inf'):
                count = ((abs_deltas >= lo)).sum()
            else:
                count = ((abs_deltas >= lo) & (abs_deltas < hi)).sum()
            pct = 100 * count / len(abs_deltas)
            print(f"    {name:>8}: {count:>7,} ({pct:5.1f}%)")

    # ==========================================================================
    # Active vs Passive Thermal Modification
    # ==========================================================================

    print("\n" + "=" * 70)
    print("ACTIVE vs PASSIVE THERMAL MODIFICATION")
    print("=" * 70)
    print("\nActive = HVAC fighting thermal gradient (outdoor works against setpoint)")
    print("Passive = Thermal gradient helps (outdoor works toward setpoint)")

    total_active_heating = sum(r["active_heating"] for r in results)
    total_passive_heating = sum(r["passive_heating"] for r in results)
    total_active_cooling = sum(r["active_cooling"] for r in results)
    total_passive_cooling = sum(r["passive_cooling"] for r in results)
    total_passive_mode = sum(r["passive_mode"] for r in results)

    total = total_active_heating + total_passive_heating + total_active_cooling + total_passive_cooling + total_passive_mode

    print(f"\n### HEATING MODE (indoor < heat_setpoint)")
    total_heating = total_active_heating + total_passive_heating
    if total_heating > 0:
        print(f"  Total samples: {total_heating:,} ({100*total_heating/total:.1f}% of all)")
        print(f"  Active (outdoor < indoor, HVAC fighting cold): {total_active_heating:,} ({100*total_active_heating/total_heating:.1f}%)")
        print(f"  Passive (outdoor > indoor, warmth helps):      {total_passive_heating:,} ({100*total_passive_heating/total_heating:.1f}%)")

    print(f"\n### COOLING MODE (indoor > cool_setpoint)")
    total_cooling = total_active_cooling + total_passive_cooling
    if total_cooling > 0:
        print(f"  Total samples: {total_cooling:,} ({100*total_cooling/total:.1f}% of all)")
        print(f"  Active (outdoor > indoor, HVAC fighting heat): {total_active_cooling:,} ({100*total_active_cooling/total_cooling:.1f}%)")
        print(f"  Passive (outdoor < indoor, coolness helps):    {total_passive_cooling:,} ({100*total_passive_cooling/total_cooling:.1f}%)")

    print(f"\n### PASSIVE MODE (within deadband)")
    print(f"  Total samples: {total_passive_mode:,} ({100*total_passive_mode/total:.1f}% of all)")

    print(f"\n### SUMMARY")
    print(f"  HVAC actively working:     {total_active_heating + total_active_cooling:,} ({100*(total_active_heating + total_active_cooling)/total:.1f}%)")
    print(f"  Thermal gradient helping:  {total_passive_heating + total_passive_cooling:,} ({100*(total_passive_heating + total_passive_cooling)/total:.1f}%)")
    print(f"  Passive (in deadband):     {total_passive_mode:,} ({100*total_passive_mode/total:.1f}%)")

    # ==========================================================================
    # Homes with useful data for incident modeling
    # ==========================================================================

    print("\n" + "=" * 70)
    print("HOMES SUITABLE FOR INCIDENT-BASED MODELING")
    print("=" * 70)

    # Homes with at least N setpoint changes
    thresholds = [10, 50, 100, 200, 500]
    for thresh in thresholds:
        suitable = sum(1 for c in total_change_counts if c >= thresh)
        total_changes_in_suitable = sum(c for c in total_change_counts if c >= thresh)
        print(f"  Homes with ≥{thresh:>3} changes: {suitable:>4} homes, {total_changes_in_suitable:,} total changes")

    # Top homes by changes
    print(f"\n### TOP 20 HOMES BY SETPOINT CHANGES")
    sorted_results = sorted(results, key=lambda r: r["n_heat_changes"] + r["n_cool_changes"], reverse=True)
    print(f"{'Home':<20} {'Heat':>8} {'Cool':>8} {'Total':>8}")
    print("-" * 50)
    for r in sorted_results[:20]:
        total = r["n_heat_changes"] + r["n_cool_changes"]
        print(f"{r['home_id']:<20} {r['n_heat_changes']:>8} {r['n_cool_changes']:>8} {total:>8}")


if __name__ == "__main__":
    main()
