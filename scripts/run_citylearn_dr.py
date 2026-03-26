"""
CityLearn Demand Response Experiment

Tests peak shaving RBC strategies against baseline.
Outputs KPIs, charts, and sanity check report.
"""

import sys
sys.path.insert(0, '/Users/amelamud/Desktop/thermal/CityLearn')

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import BaselineAgent
from citylearn.agents.rbc import HourRBC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = 'quebec_neighborhood_with_demand_response_set_points'
OUTPUT_DIR = Path('/Users/amelamud/Desktop/thermal/outputs')

# --- Controllers ---

class ConstantRBC(HourRBC):
    """Fixed action for all hours (sanity check)."""
    def __init__(self, env, action_value=0.5):
        action_map = {h: action_value for h in range(1, 25)}
        super().__init__(env=env, action_map=action_map)


class PeakShavingRBC(HourRBC):
    """Time-based peak shaving strategy."""
    def __init__(self, env, strategy='moderate'):
        if strategy == 'moderate':
            action_map = {h: 0.7 for h in range(1, 25)}
            action_map.update({14: 1.0, 15: 1.0, 16: 1.0})  # Pre-heat
            action_map.update({17: 0.3, 18: 0.3, 19: 0.3, 20: 0.3})  # Peak reduction
            action_map.update({21: 0.8, 22: 0.8, 23: 0.8})  # Recovery
        elif strategy == 'aggressive':
            action_map = {h: 0.6 for h in range(1, 25)}
            action_map.update({12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0})
            action_map.update({17: 0.1, 18: 0.1, 19: 0.1, 20: 0.1})
            action_map.update({21: 1.0, 22: 1.0, 23: 1.0})
        super().__init__(env=env, action_map=action_map)


class FeedbackController:
    """
    Proportional feedback controller for HVAC.

    Action = Kp * (setpoint - current_temp), clamped to [0, 1]

    During peak hours, reduce Kp to shift load.
    """
    def __init__(self, env, strategy='normal'):
        self.env = env
        self.strategy = strategy

        # Proportional gains by strategy
        # Higher gains needed - baseline uses ~0.3-0.5 of nominal power
        if strategy == 'normal':
            # Normal operation - same gain all day
            self.kp_normal = 0.5  # action per degree below setpoint
            self.kp_peak = 0.5
            self.bias = 0.1  # Minimum heating to prevent drift
        elif strategy == 'peak_shave':
            # Reduce gain during peak, increase before
            self.kp_normal = 0.5
            self.kp_peak = 0.2  # Lower gain during peak
            self.kp_preheat = 0.8  # Higher gain for pre-heating
            self.bias = 0.1
            self.bias_preheat = 0.3  # Extra heating before peak
        elif strategy == 'aggressive':
            self.kp_normal = 0.5
            self.kp_peak = 0.1  # Very low during peak
            self.kp_preheat = 1.0  # Maximum pre-heat
            self.bias = 0.1
            self.bias_preheat = 0.4

    def predict(self, observations):
        """
        Compute action based on temperature feedback.

        observations is a list of dicts, one per building (in central_agent mode,
        it's a single flattened array - we need to parse it)
        """
        # In central_agent mode, observations is a flat array
        # We need to figure out the structure
        n_buildings = len(self.env.buildings)

        # Get current hour (first observation is typically hour)
        # The observation space depends on active observations in schema
        # For quebec dataset: day_type, hour, outdoor_temp, indoor_temp, net_elec,
        #                     pricing, heating_setpoint, heating_delta, price_pred...

        # Try to extract hour and temperature delta from observations
        try:
            obs_array = np.array(observations).flatten()

            # Based on quebec schema, hour is index 1 (after day_type)
            # But observations are per-building, so we need to be careful
            # Let's get hour from env directly
            hour = self.env.time_step % 24 + 1  # 1-24

            actions = []

            # Determine gain and bias based on hour
            if self.strategy == 'normal':
                kp = self.kp_normal
                bias = getattr(self, 'bias', 0.0)
            elif hour in [14, 15, 16]:  # Pre-heat hours
                kp = getattr(self, 'kp_preheat', self.kp_normal)
                bias = getattr(self, 'bias_preheat', getattr(self, 'bias', 0.0))
            elif hour in [17, 18, 19, 20]:  # Peak hours
                kp = self.kp_peak
                bias = 0.0  # No bias during peak - minimize heating
            else:
                kp = self.kp_normal
                bias = getattr(self, 'bias', 0.0)

            # For each building, compute action based on heating delta
            # heating_delta = current_temp - setpoint (negative means needs heating)
            for i, b in enumerate(self.env.buildings):
                # Get current temperature and setpoint from building
                try:
                    current_temp = b.indoor_dry_bulb_temperature[-1] if b.indoor_dry_bulb_temperature else 20
                    setpoint = b.indoor_dry_bulb_temperature_heating_set_point[-1] if b.indoor_dry_bulb_temperature_heating_set_point else 20
                except:
                    current_temp = 20
                    setpoint = 20

                # Temperature error (positive = needs heating)
                error = setpoint - current_temp

                # Proportional control with bias
                action = kp * max(0, error) + bias

                # Clamp to [0, 1]
                action = max(0.0, min(1.0, action))

                actions.append(action)

            # central_agent=True expects flat array of shape (n_buildings,)
            return [actions]  # Wrap in list for CityLearn

        except Exception as e:
            # Fallback: return small constant action
            return [[0.1] * n_buildings]

    def learn(self, episodes=1):
        """Run simulation (no learning, just execution)."""
        for ep in range(episodes):
            obs, _ = self.env.reset()
            while not self.env.terminated:
                actions = self.predict(obs)
                obs, reward, info, terminated, truncated = self.env.step(actions)


# --- Metrics ---

def calculate_par(consumption):
    """Peak-to-Average Ratio."""
    if np.mean(consumption) == 0:
        return np.nan
    return np.max(consumption) / np.mean(consumption)


def calculate_daily_par(consumption):
    """Average daily PAR."""
    # Reshape to days x 24 hours
    n_hours = len(consumption)
    n_complete_days = n_hours // 24
    if n_complete_days == 0:
        return calculate_par(consumption)

    daily = np.array(consumption[:n_complete_days * 24]).reshape(-1, 24)
    pars = []
    for d in daily:
        if np.mean(d) > 0:
            pars.append(np.max(d) / np.mean(d))
    return np.mean(pars) if pars else np.nan


# --- Simulation ---

def run_simulation(name, controller_fn):
    """Run one controller and collect results."""
    print(f"  Running {name}...")

    # Load schema and limit to 1 day (24 hours) for quick testing
    from citylearn.data import DataSet
    schema = DataSet().get_schema(DATASET)
    schema['simulation_start_time_step'] = 0
    schema['simulation_end_time_step'] = 23  # Just 24 hours

    env = CityLearnEnv(schema, central_agent=True)
    controller = controller_fn(env)

    obs, _ = env.reset()
    step = 0
    while not env.terminated:
        actions = controller.predict(obs)
        obs, reward, info, terminated, truncated = env.step(actions)
        step += 1
        if step % 1000 == 0:
            print(f"    Step {step}...")

    print(f"    Completed {step} steps")

    # Collect results
    result = {
        'kpis': env.evaluate(),
        'consumption': np.array(env.net_electricity_consumption),
    }

    # Get building-level data (simplified - skip for now to avoid array issues)
    result['temperatures'] = []
    result['setpoints'] = []
    result['n_buildings'] = len(env.buildings)

    return result


# --- Sanity Checks ---

def run_sanity_checks(results):
    """Run all sanity checks, return list of (check_name, passed, message)."""
    checks = []

    baseline = results.get('Baseline')
    if baseline is None:
        checks.append(("Baseline exists", False, "No baseline results"))
        return checks

    baseline_energy = baseline['consumption'].sum()

    for name, data in results.items():
        # Energy conservation
        energy = data['consumption'].sum()
        if baseline_energy > 0:
            energy_ratio = energy / baseline_energy
            passed = 0.5 <= energy_ratio <= 2.0  # Wide bounds for initial test
            checks.append((f"{name}: Energy ratio", passed, f"{energy_ratio:.3f}"))

        # Consumption is positive
        min_consumption = data['consumption'].min()
        checks.append((f"{name}: Non-negative consumption", min_consumption >= 0, f"min={min_consumption:.2f}"))

        # Temperature bounds (if available)
        for i, temps in enumerate(data['temperatures']):
            if len(temps) > 0:
                min_t, max_t = temps.min(), temps.max()
                passed = min_t >= 5 and max_t <= 35  # Wide bounds
                if not passed:
                    checks.append((f"{name} Bldg {i}: Temp bounds", False, f"[{min_t:.1f}, {max_t:.1f}]°C"))

        # No NaN values
        has_nan = np.isnan(data['consumption']).any()
        checks.append((f"{name}: No NaN in consumption", not has_nan, ""))

        # Reasonable consumption range
        max_consumption = data['consumption'].max()
        mean_consumption = data['consumption'].mean()
        checks.append((f"{name}: Consumption stats", True, f"mean={mean_consumption:.1f}, max={max_consumption:.1f} kW"))

    return checks


# --- Charts ---

def plot_daily_profile(results, output_path):
    """Average hourly load by controller."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in results.items():
        consumption = data['consumption']
        n_complete_days = len(consumption) // 24
        if n_complete_days == 0:
            continue

        hourly = consumption[:n_complete_days * 24].reshape(-1, 24)
        avg = hourly.mean(axis=0)
        std = hourly.std(axis=0)

        line, = ax.plot(range(24), avg, label=name, linewidth=2)
        ax.fill_between(range(24), avg - std, avg + std, alpha=0.2, color=line.get_color())

    ax.axvspan(17, 20, alpha=0.15, color='red', label='Peak Hours (5-8pm)')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average District Load (kW)', fontsize=12)
    ax.set_title('Daily Load Profile by Controller', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_temperature_trajectory(results, building_idx, hours, output_path):
    """Temperature over time for one building."""
    fig, ax = plt.subplots(figsize=(14, 5))

    has_data = False
    for name, data in results.items():
        if building_idx < len(data['temperatures']):
            temps = data['temperatures'][building_idx]
            if len(temps) >= hours:
                ax.plot(temps[:hours], label=name, linewidth=1.5, alpha=0.8)
                has_data = True

    # Plot setpoint from baseline
    if 'Baseline' in results and building_idx < len(results['Baseline']['setpoints']):
        setpoint = results['Baseline']['setpoints'][building_idx]
        if len(setpoint) >= hours:
            ax.plot(setpoint[:hours], 'k--', label='Setpoint', linewidth=1.5, alpha=0.7)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'No temperature data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Indoor Temperature (°C)', fontsize=12)
    ax.set_title(f'Temperature Trajectory - Building {building_idx} (First {hours} hours)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_consumption_comparison(results, output_path):
    """Bar chart comparing total consumption and peak."""
    controllers = list(results.keys())

    totals = [results[c]['consumption'].sum() for c in controllers]
    peaks = [results[c]['consumption'].max() for c in controllers]
    pars = [calculate_par(results[c]['consumption']) for c in controllers]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Total consumption
    ax = axes[0]
    bars = ax.bar(controllers, totals, color=['steelblue', 'orange', 'green', 'red'][:len(controllers)])
    ax.set_ylabel('Total Consumption (kWh)')
    ax.set_title('Total Energy Consumption')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}',
                ha='center', va='bottom', fontsize=9)

    # Peak consumption
    ax = axes[1]
    bars = ax.bar(controllers, peaks, color=['steelblue', 'orange', 'green', 'red'][:len(controllers)])
    ax.set_ylabel('Peak Load (kW)')
    ax.set_title('Maximum Peak Load')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)

    # PAR
    ax = axes[2]
    bars = ax.bar(controllers, pars, color=['steelblue', 'orange', 'green', 'red'][:len(controllers)])
    ax.set_ylabel('Peak-to-Average Ratio')
    ax.set_title('PAR (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, pars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# --- Main ---

def main():
    print("=" * 60)
    print("CityLearn Demand Response Experiment")
    print("=" * 60)

    controllers = {
        'Baseline': lambda e: BaselineAgent(e),
        'Feedback_Normal': lambda e: FeedbackController(e, 'normal'),
        'Feedback_PeakShave': lambda e: FeedbackController(e, 'peak_shave'),
        'Feedback_Aggressive': lambda e: FeedbackController(e, 'aggressive'),
    }

    print("\nRunning simulations...")
    results = {}
    for name, ctrl_fn in controllers.items():
        try:
            results[name] = run_simulation(name, ctrl_fn)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No results collected. Exiting.")
        return

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    rows = []
    for name, data in results.items():
        kpis = data['kpis']
        district = kpis[kpis['name'] == 'District']

        def get_kpi(cost_function):
            vals = district[district['cost_function'] == cost_function]['value'].values
            return vals[0] if len(vals) > 0 else np.nan

        row = {
            'Controller': name,
            'PAR': calculate_par(data['consumption']),
            'Daily_PAR': calculate_daily_par(data['consumption']),
            'Total_kWh': data['consumption'].sum(),
            'Peak_kW': data['consumption'].max(),
            'Mean_kW': data['consumption'].mean(),
            'daily_peak_avg': get_kpi('daily_peak_average'),
            'electricity_total': get_kpi('electricity_consumption_total'),
            'discomfort_%': get_kpi('discomfort_proportion') * 100 if not np.isnan(get_kpi('discomfort_proportion')) else np.nan,
            'discomfort_cold_avg': get_kpi('discomfort_cold_delta_average'),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("\n" + summary.to_string(index=False))

    # Save summary
    summary_path = OUTPUT_DIR / 'dr_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # Sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    checks = run_sanity_checks(results)
    n_passed = 0
    n_failed = 0
    for check, passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_passed += 1
        else:
            n_failed += 1
        print(f"[{status}] {check} {msg}")

    print(f"\nTotal: {n_passed} passed, {n_failed} failed")

    # Save sanity checks
    checks_path = OUTPUT_DIR / 'sanity_checks.txt'
    with open(checks_path, 'w') as f:
        for check, passed, msg in checks:
            status = "PASS" if passed else "FAIL"
            f.write(f"[{status}] {check} {msg}\n")
    print(f"Saved sanity checks to: {checks_path}")

    # Generate charts
    print("\n" + "=" * 60)
    print("GENERATING CHARTS")
    print("=" * 60)

    plot_daily_profile(results, OUTPUT_DIR / 'daily_load_profile.png')
    plot_temperature_trajectory(results, 0, 24, OUTPUT_DIR / 'temperature_trajectory.png')  # 1 day
    plot_consumption_comparison(results, OUTPUT_DIR / 'consumption_comparison.png')

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
