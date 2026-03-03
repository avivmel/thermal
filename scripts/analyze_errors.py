"""
Error analysis for time-to-target prediction.

Analyzes where the model fails to identify systematic patterns.
"""

import polars as pl
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm
import time

DATA_PATH = "data/setpoint_responses.parquet"


def create_episode_samples(df: pl.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create episode-level samples (same as run_improved_baselines.py)."""
    print(f"Creating episode-level samples...")
    t0 = time.time()
    np.random.seed(42)

    records = []
    homes = df["home_id"].unique().to_list()

    for home in tqdm(homes, desc="  Processing homes"):
        home_df = df.filter(pl.col("home_id") == home)
        episodes = home_df["episode_id"].unique().to_list()

        if len(episodes) < 2:
            continue

        np.random.shuffle(episodes)
        n_train = max(1, int(len(episodes) * train_frac))
        train_episodes = set(episodes[:n_train])

        for ep_id in episodes:
            ep = home_df.filter(pl.col("episode_id") == ep_id).sort("timestep_idx")
            if len(ep) < 1:
                continue

            n_steps = len(ep)
            duration_min = n_steps * 5
            if duration_min < 5:
                continue

            first_row = ep[0]
            start_temp = first_row["Indoor_AverageTemperature"].item()
            target_sp = first_row["target_setpoint"].item()
            outdoor_temp = first_row["Outdoor_Temperature"].item()
            initial_gap = first_row["initial_gap"].item()
            change_type = first_row["change_type"].item()
            state = first_row["state"].item()
            timestamp = first_row["timestamp"].item()
            heat_runtime = first_row["HeatingEquipmentStage1_RunTime"].item()
            cool_runtime = first_row["CoolingEquipmentStage1_RunTime"].item()
            indoor_humidity = first_row["Indoor_Humidity"].item()
            outdoor_humidity = first_row["Outdoor_Humidity"].item()

            if pd.isna(start_temp) or pd.isna(outdoor_temp) or pd.isna(initial_gap):
                continue

            thermal_drive = outdoor_temp - start_temp
            if change_type == "heat_increase":
                signed_thermal_drive = thermal_drive
            else:
                signed_thermal_drive = -thermal_drive

            hour = timestamp.hour
            month = timestamp.month
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            heat_runtime = 0 if pd.isna(heat_runtime) else heat_runtime
            cool_runtime = 0 if pd.isna(cool_runtime) else cool_runtime
            system_running = 1 if (heat_runtime > 0 or cool_runtime > 0) else 0

            records.append({
                "home_id": home,
                "episode_id": ep_id,
                "is_train": ep_id in train_episodes,
                "state": state,
                "change_type": change_type,
                "duration_min": duration_min,
                "log_duration": np.log(duration_min),
                "initial_gap": initial_gap,
                "abs_gap": np.abs(initial_gap),
                "log_gap": np.log(np.abs(initial_gap) + 0.1),
                "start_temp": start_temp,
                "target_setpoint": target_sp,
                "outdoor_temp": outdoor_temp,
                "thermal_drive": thermal_drive,
                "signed_thermal_drive": signed_thermal_drive,
                "hour": hour,
                "month": month,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "indoor_humidity": indoor_humidity if not pd.isna(indoor_humidity) else 50,
                "outdoor_humidity": outdoor_humidity if not pd.isna(outdoor_humidity) else 50,
                "system_running": system_running,
                "is_heating": 1 if change_type == "heat_increase" else 0,
            })

    all_df = pd.DataFrame(records)
    train_df = all_df[all_df["is_train"]].copy()
    test_df = all_df[~all_df["is_train"]].copy()

    print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")
    return train_df, test_df


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Fit the best model and add predictions to test_df."""
    print("Fitting hierarchical + enhanced model...")

    feature_cols = [
        "log_gap", "is_heating", "signed_thermal_drive",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "system_running", "outdoor_humidity"
    ]

    train_data = train_df.copy()
    test_data = test_df.copy()

    for col in feature_cols:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)

    X_train = train_data[feature_cols].values
    y_train = train_data["log_duration"].values

    global_model = Ridge(alpha=1.0)
    global_model.fit(X_train, y_train)

    global_pred = global_model.predict(X_train)
    residuals = y_train - global_pred

    # Per-home + mode offsets
    home_mode_offsets = {}
    for home in train_data["home_id"].unique():
        for mode in [0, 1]:
            mask = (train_data["home_id"] == home) & (train_data["is_heating"] == mode)
            if mask.sum() >= 3:
                offset = residuals[mask].mean()
                n = mask.sum()
                weight = n / (n + 10)
                home_mode_offsets[(home, mode)] = offset * weight

    # Predict
    X_test = test_data[feature_cols].values
    global_pred_test = global_model.predict(X_test)

    predictions = []
    for i, (idx, row) in enumerate(test_data.iterrows()):
        home = row["home_id"]
        mode = int(row["is_heating"])
        offset = home_mode_offsets.get((home, mode), 0.0)
        log_pred = global_pred_test[i] + offset
        predictions.append(np.exp(log_pred))

    test_data = test_data.copy()
    test_data["pred_duration"] = predictions
    test_data["error"] = test_data["pred_duration"] - test_data["duration_min"]
    test_data["abs_error"] = np.abs(test_data["error"])
    test_data["pct_error"] = 100 * test_data["abs_error"] / test_data["duration_min"]

    return test_data


def analyze_errors(df: pd.DataFrame):
    """Comprehensive error analysis."""

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    # Overall stats
    print("\n## Overall Error Distribution")
    print(f"  MAE: {df['abs_error'].mean():.1f} min")
    print(f"  Median AE: {df['abs_error'].median():.1f} min")
    print(f"  Std: {df['abs_error'].std():.1f} min")
    print(f"  Bias: {df['error'].mean():+.1f} min")
    print(f"  P90 error: {df['abs_error'].quantile(0.90):.1f} min")
    print(f"  P95 error: {df['abs_error'].quantile(0.95):.1f} min")
    print(f"  P99 error: {df['abs_error'].quantile(0.99):.1f} min")

    # Error percentiles
    print("\n## Error Percentile Distribution")
    for pct in [50, 75, 90, 95, 99]:
        val = df['abs_error'].quantile(pct/100)
        print(f"  P{pct}: {val:.1f} min")

    # Worst errors
    print("\n## Worst 20 Predictions")
    worst = df.nlargest(20, 'abs_error')[['duration_min', 'pred_duration', 'error', 'abs_gap',
                                           'change_type', 'system_running', 'state', 'outdoor_temp']]
    print(worst.to_string())

    # Systematic patterns
    print("\n" + "=" * 80)
    print("SYSTEMATIC PATTERNS")
    print("=" * 80)

    # By actual duration (are long episodes hard?)
    print("\n## Error by Actual Duration")
    df['duration_bucket'] = pd.cut(df['duration_min'],
                                    bins=[0, 15, 30, 60, 120, 240, 1000],
                                    labels=['<15min', '15-30', '30-60', '60-120', '120-240', '>240'])
    for bucket in df['duration_bucket'].cat.categories:
        subset = df[df['duration_bucket'] == bucket]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            n = len(subset)
            print(f"  {bucket:>10}: MAE={mae:>6.1f}, Bias={bias:>+6.1f}, n={n:>6,}")

    # By gap size
    print("\n## Error by Initial Gap")
    df['gap_bucket'] = pd.cut(df['abs_gap'],
                               bins=[0, 1, 2, 3, 5, 10, 20],
                               labels=['0-1°F', '1-2°F', '2-3°F', '3-5°F', '5-10°F', '>10°F'])
    for bucket in df['gap_bucket'].cat.categories:
        subset = df[df['gap_bucket'] == bucket]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            n = len(subset)
            print(f"  {bucket:>10}: MAE={mae:>6.1f}, Bias={bias:>+6.1f}, n={n:>6,}")

    # By change type
    print("\n## Error by Change Type")
    for ctype in ['heat_increase', 'cool_decrease']:
        subset = df[df['change_type'] == ctype]
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        print(f"  {ctype:>15}: MAE={mae:.1f}, Bias={bias:+.1f}, n={len(subset):,}")

    # By system_running
    print("\n## Error by System Running at Start")
    for running in [0, 1]:
        subset = df[df['system_running'] == running]
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        label = "Running" if running else "Not running"
        print(f"  {label:>12}: MAE={mae:.1f}, Bias={bias:+.1f}, n={len(subset):,}")

    # By state
    print("\n## Error by State")
    for state in df['state'].unique():
        subset = df[df['state'] == state]
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        print(f"  {state:>4}: MAE={mae:.1f}, Bias={bias:+.1f}, n={len(subset):,}")

    # By month
    print("\n## Error by Month")
    for month in sorted(df['month'].unique()):
        subset = df[df['month'] == month]
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        print(f"  {month:>2}: MAE={mae:.1f}, Bias={bias:+.1f}, n={len(subset):,}")

    # By hour
    print("\n## Error by Hour (grouped)")
    df['hour_bucket'] = pd.cut(df['hour'],
                                bins=[-1, 6, 12, 18, 24],
                                labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'])
    for bucket in df['hour_bucket'].cat.categories:
        subset = df[df['hour_bucket'] == bucket]
        mae = subset['abs_error'].mean()
        bias = subset['error'].mean()
        print(f"  {bucket:>20}: MAE={mae:.1f}, Bias={bias:+.1f}, n={len(subset):,}")

    # By outdoor temp
    print("\n## Error by Outdoor Temperature")
    df['outdoor_bucket'] = pd.cut(df['outdoor_temp'],
                                   bins=[-20, 32, 50, 70, 85, 120],
                                   labels=['<32°F', '32-50°F', '50-70°F', '70-85°F', '>85°F'])
    for bucket in df['outdoor_bucket'].cat.categories:
        subset = df[df['outdoor_bucket'] == bucket]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            print(f"  {bucket:>10}: MAE={mae:.1f}, Bias={bias:+.1f}, n={len(subset):,}")

    # Per-home analysis
    print("\n" + "=" * 80)
    print("PER-HOME ANALYSIS")
    print("=" * 80)

    home_stats = df.groupby('home_id').agg({
        'abs_error': ['mean', 'std', 'count'],
        'error': 'mean',
        'duration_min': 'mean'
    }).reset_index()
    home_stats.columns = ['home_id', 'mae', 'std', 'n_episodes', 'bias', 'avg_duration']

    print("\n## Home Error Distribution")
    print(f"  Home MAE - Min: {home_stats['mae'].min():.1f}, Max: {home_stats['mae'].max():.1f}")
    print(f"  Home MAE - P25: {home_stats['mae'].quantile(0.25):.1f}, Median: {home_stats['mae'].median():.1f}, P75: {home_stats['mae'].quantile(0.75):.1f}")

    # Worst homes
    print("\n## Hardest-to-Predict Homes (top 10 by MAE, min 10 episodes)")
    worst_homes = home_stats[home_stats['n_episodes'] >= 10].nlargest(10, 'mae')
    print(worst_homes.to_string(index=False))

    # Best homes
    print("\n## Easiest-to-Predict Homes (top 10 by MAE, min 10 episodes)")
    best_homes = home_stats[home_stats['n_episodes'] >= 10].nsmallest(10, 'mae')
    print(best_homes.to_string(index=False))

    # What makes hard homes hard?
    print("\n## Characteristics of Hard vs Easy Homes")
    hard_home_ids = set(worst_homes['home_id'])
    easy_home_ids = set(best_homes['home_id'])

    hard_episodes = df[df['home_id'].isin(hard_home_ids)]
    easy_episodes = df[df['home_id'].isin(easy_home_ids)]

    print("\n  Feature comparison:")
    for col in ['abs_gap', 'duration_min', 'outdoor_temp', 'system_running']:
        hard_mean = hard_episodes[col].mean()
        easy_mean = easy_episodes[col].mean()
        print(f"    {col:>20}: Hard={hard_mean:>8.1f}, Easy={easy_mean:>8.1f}")

    # Error vs predicted (calibration)
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)

    print("\n## Error by Predicted Duration")
    df['pred_bucket'] = pd.cut(df['pred_duration'],
                                bins=[0, 20, 40, 60, 100, 200, 1000],
                                labels=['<20min', '20-40', '40-60', '60-100', '100-200', '>200'])
    for bucket in df['pred_bucket'].cat.categories:
        subset = df[df['pred_bucket'] == bucket]
        if len(subset) > 0:
            mae = subset['abs_error'].mean()
            bias = subset['error'].mean()
            actual_mean = subset['duration_min'].mean()
            pred_mean = subset['pred_duration'].mean()
            print(f"  {bucket:>10}: MAE={mae:>6.1f}, Bias={bias:>+7.1f}, "
                  f"Actual={actual_mean:>6.1f}, Pred={pred_mean:>6.1f}, n={len(subset):>5,}")

    # Underprediction vs overprediction
    print("\n## Under vs Over Prediction")
    under = df[df['error'] < 0]  # Predicted too short
    over = df[df['error'] > 0]   # Predicted too long
    print(f"  Underpredicted (pred < actual): {len(under):,} ({100*len(under)/len(df):.1f}%)")
    print(f"    Mean error: {under['error'].mean():.1f} min")
    print(f"  Overpredicted (pred > actual):  {len(over):,} ({100*len(over)/len(df):.1f}%)")
    print(f"    Mean error: {over['error'].mean():.1f} min")

    # Large errors investigation
    print("\n" + "=" * 80)
    print("LARGE ERROR INVESTIGATION")
    print("=" * 80)

    large_errors = df[df['abs_error'] > df['abs_error'].quantile(0.95)]
    print(f"\n## Episodes with >P95 Error ({len(large_errors):,} episodes)")

    print("\n  Breakdown by duration:")
    for bucket in large_errors['duration_bucket'].value_counts().index:
        n = (large_errors['duration_bucket'] == bucket).sum()
        pct = 100 * n / len(large_errors)
        print(f"    {bucket}: {n:,} ({pct:.1f}%)")

    print("\n  Breakdown by change type:")
    for ctype in large_errors['change_type'].value_counts().index:
        n = (large_errors['change_type'] == ctype).sum()
        pct = 100 * n / len(large_errors)
        baseline_pct = 100 * (df['change_type'] == ctype).sum() / len(df)
        print(f"    {ctype}: {n:,} ({pct:.1f}%) [baseline: {baseline_pct:.1f}%]")

    print("\n  Breakdown by system_running:")
    for running in [0, 1]:
        n = (large_errors['system_running'] == running).sum()
        pct = 100 * n / len(large_errors)
        baseline_pct = 100 * (df['system_running'] == running).sum() / len(df)
        label = "Running" if running else "Not running"
        print(f"    {label}: {n:,} ({pct:.1f}%) [baseline: {baseline_pct:.1f}%]")

    # Very long episodes analysis
    print("\n## Very Long Episodes (>2 hours)")
    long = df[df['duration_min'] > 120]
    print(f"  Count: {len(long):,} ({100*len(long)/len(df):.1f}% of all)")
    print(f"  MAE: {long['abs_error'].mean():.1f} min")
    print(f"  Bias: {long['error'].mean():+.1f} min")
    print(f"  They account for {100*long['abs_error'].sum()/df['abs_error'].sum():.1f}% of total absolute error")

    # Very short episodes analysis
    print("\n## Very Short Episodes (<15 min)")
    short = df[df['duration_min'] < 15]
    print(f"  Count: {len(short):,} ({100*len(short)/len(df):.1f}% of all)")
    print(f"  MAE: {short['abs_error'].mean():.1f} min")
    print(f"  Bias: {short['error'].mean():+.1f} min")

    return df


def main():
    print("Loading data...")
    df = pl.read_parquet(DATA_PATH)

    train_df, test_df = create_episode_samples(df)
    test_with_pred = fit_and_predict(train_df, test_df)
    analyze_errors(test_with_pred)


if __name__ == "__main__":
    main()
