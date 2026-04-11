from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rule-preheat MPC demo outputs.")
    parser.add_argument("--history", default="outputs/rule_preheat_mpc_histories.csv")
    parser.add_argument("--output", default="outputs/rule_preheat_mpc_trajectory.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    history_path = Path(args.history)
    output_path = Path(args.output)
    df = pd.read_csv(history_path, parse_dates=["timestamp"])
    if df.empty:
        raise ValueError(f"No rows found in {history_path}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for policy, policy_df in df.groupby("policy", sort=False):
        x = range(len(policy_df))
        axes[0].plot(x, policy_df["indoor_temp"], label=f"{policy} indoor")
        axes[0].step(
            x,
            policy_df["heat_setpoint"],
            where="post",
            linestyle="--",
            alpha=0.75,
            label=f"{policy} heat sp",
        )
        axes[1].plot(x, policy_df["hvac_power_kw"], label=policy)

    axes[0].set_ylabel("Temperature (F)")
    axes[0].legend(loc="best", ncols=2, fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_ylabel("HVAC power (kW)")
    axes[1].set_xlabel("5-minute step")
    axes[1].legend(loc="best", ncols=2, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
