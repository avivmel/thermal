# CityLearn Demand Response Experiment Plan

## Objective

Test a simple rule-based peak shaving strategy for residential HVAC demand response using CityLearn's Quebec dataset. Compare against baseline (no control) to measure PAR reduction and comfort impacts.

## Dataset

**`quebec_neighborhood_with_demand_response_set_points`**

| Property | Value |
|----------|-------|
| Buildings | 20 Quebec single-family homes |
| Time resolution | 1 hour |
| Simulation period | 6,480 timesteps (~270 days, winter heating season) |
| Episodes | 3 x 90 days each |
| HVAC | Heat pumps (14-47 kW nominal power) |
| Dynamics | Per-building LSTM thermal models |
| Storage | None (pure HVAC control) |

### Action Space

Single continuous action per building: `heating_device ∈ [0, 1]`
- `0.0` = heat pump off
- `1.0` = heat pump at full nominal power

### Baseline vs Controlled Mode

| Mode | Action | What Happens |
|------|--------|--------------|
| **Baseline** | `None` | Heat pump delivers pre-computed `heating_demand` from EnergyPlus CSV. Temperature perfectly tracks setpoint. |
| **Controlled** | `0.0 - 1.0` | Heat pump delivers `action × nominal_power`. LSTM dynamics model simulates resulting indoor temperature. |

### Key Observations Available

| Observation | Description |
|-------------|-------------|
| `hour` | Hour of day (1-24) |
| `indoor_dry_bulb_temperature` | Current indoor temp (°C) |
| `outdoor_dry_bulb_temperature` | Current outdoor temp (°C) |
| `indoor_dry_bulb_temperature_heating_set_point` | Target setpoint (°C) |
| `indoor_dry_bulb_temperature_heating_delta` | Current temp - setpoint |
| `electricity_pricing` | Current electricity price |
| `electricity_pricing_predicted_{1,2,3}` | Price forecasts |

---

## Experiment Design

### Controllers to Test

#### 1. Baseline (No Control)
- Use `BaselineAgent` which outputs action = `None`
- Heat pump delivers pre-computed ideal load from CSV
- Represents "business as usual" thermostat behavior

#### 2. Constant Output (Sanity Check)
- Fixed action = 0.5 for all hours
- Verifies that actions actually affect consumption and temperature
- Should show different consumption than baseline

#### 3. Peak Shaving RBC (Moderate)
Simple time-based rule:
```
Hours 14-16: action = 1.0  (pre-heat before peak)
Hours 17-20: action = 0.3  (reduce during peak, 5-8pm)
Hours 21-23: action = 0.8  (recovery)
Other hours: action = 0.7  (normal operation)
```

#### 4. Aggressive Peak Shaving
More extreme reduction:
```
Hours 12-16: action = 1.0  (extended pre-heat)
Hours 17-20: action = 0.1  (aggressive reduction)
Hours 21-23: action = 1.0  (aggressive recovery)
Other hours: action = 0.6  (slightly reduced baseline)
```

---

## Metrics & KPIs

### Primary: Peak-to-Average Ratio (PAR)

```
PAR = max(hourly_load) / mean(hourly_load)
```

Calculate at:
- District level (sum of all buildings)
- Per-building level
- Daily average PAR

### Secondary: CityLearn Built-in KPIs

| KPI | Description | Target |
|-----|-------------|--------|
| `daily_peak_average` | Avg daily peak / baseline | < 1.0 |
| `all_time_peak_average` | Max peak / baseline | < 1.0 |
| `ramping_average` | Load variability / baseline | < 1.0 |
| `electricity_consumption_total` | Total energy / baseline | ≈ 1.0 (shouldn't increase much) |
| `cost_total` | Electricity cost / baseline | < 1.0 (if TOU pricing) |

### Comfort Metrics

| KPI | Description | Constraint |
|-----|-------------|------------|
| `discomfort_cold_proportion` | % time below setpoint | Should be low |
| `discomfort_cold_delta_average` | Avg °C below setpoint | < 2°C acceptable |
| `discomfort_cold_delta_maximum` | Max °C below setpoint | < 5°C |
| `discomfort_hot_proportion` | % time above setpoint | Should be ~0 for heating |

---

## Sanity Checks (Critical)

### 1. Action Effect Verification
- [ ] Confirm actions actually change consumption (compare constant 0.3 vs 0.7 vs 1.0)
- [ ] Plot hourly consumption for a single building under different constant actions
- [ ] Verify consumption scales roughly with action value

### 2. Thermal Dynamics Verification
- [ ] Plot indoor temperature trajectory for one building over 72 hours
- [ ] Verify temperature drops when action is low, rises when high
- [ ] Check temperature stays within reasonable bounds (15-25°C)
- [ ] Confirm LSTM model responds sensibly to action changes

### 3. Load Profile Shape
- [ ] Plot 24-hour average load profile (should show morning/evening patterns)
- [ ] Verify peak hours in data match expected behavior
- [ ] Compare baseline load profile to controlled profiles

### 4. Energy Conservation
- [ ] Total energy with peak shaving should be similar to baseline (±15%)
- [ ] If much higher: pre-heating is excessive
- [ ] If much lower: comfort is being sacrificed

### 5. Comfort Bounds
- [ ] No indoor temps below 15°C (unsafe)
- [ ] Discomfort proportion < 20% for moderate strategies
- [ ] Discomfort delta < 3°C on average

### 6. Cross-Building Consistency
- [ ] All buildings should show similar directional changes
- [ ] Flag any building with opposite behavior (may indicate data issue)
- [ ] Check that building sizes correlate with consumption levels

### 7. Value Reasonableness
- [ ] Heat pump power draws in reasonable range (5-50 kW)
- [ ] Indoor temperatures in reasonable range (15-25°C)
- [ ] Outdoor temperatures match Quebec winter (-30 to +10°C)
- [ ] No NaN or infinite values in any output

---

## Output Artifacts

### 1. Summary Table
```
| Controller | PAR  | Daily Peak | Total Energy | Cost | Discomfort % | Max Temp Drop |
|------------|------|------------|--------------|------|--------------|---------------|
| Baseline   | X.XX | 1.000      | 1.000        | 1.00 | X.X%         | X.X°C         |
| Constant   | X.XX | X.XXX      | X.XXX        | X.XX | X.X%         | X.X°C         |
| Moderate   | X.XX | X.XXX      | X.XXX        | X.XX | X.X%         | X.X°C         |
| Aggressive | X.XX | X.XXX      | X.XXX        | X.XX | X.X%         | X.X°C         |
```

### 2. Daily Load Profile Chart
- X-axis: Hour of day (0-23)
- Y-axis: Average district load (kW)
- Lines: One per controller
- Shaded region: Peak hours (17-20)
- Purpose: Visualize load shifting effect

### 3. Temperature Trajectory Chart
- X-axis: Timestep (first 72 hours = 3 days)
- Y-axis: Indoor temperature (°C)
- Lines: One per controller for same building
- Horizontal line: Setpoint
- Purpose: Verify thermal dynamics make sense

### 4. Sanity Check Report
- List of all checks with PASS/FAIL
- Warnings for any anomalies
- Building-level outliers flagged

---

## Implementation

### Script: `scripts/run_citylearn_dr.py`

```python
"""
CityLearn Demand Response Experiment

Tests peak shaving RBC strategies against baseline.
Outputs KPIs, charts, and sanity check report.
"""

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import BaselineAgent
from citylearn.agents.rbc import HourRBC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET = 'quebec_neighborhood_with_demand_response_set_points'

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

# --- Metrics ---

def calculate_par(consumption):
    """Peak-to-Average Ratio."""
    return np.max(consumption) / np.mean(consumption)

def calculate_daily_par(consumption):
    """Average daily PAR."""
    daily = np.array(consumption).reshape(-1, 24)
    pars = [np.max(d) / np.mean(d) if np.mean(d) > 0 else np.nan for d in daily]
    return np.nanmean(pars)

# --- Simulation ---

def run_simulation(controller_fn):
    """Run one controller and collect results."""
    env = CityLearnEnv(DATASET, central_agent=True)
    controller = controller_fn(env)

    obs, _ = env.reset()
    while not env.terminated:
        actions = controller.predict(obs)
        obs, reward, info, terminated, truncated = env.step(actions)

    return {
        'kpis': env.evaluate(),
        'consumption': np.array(env.net_electricity_consumption),
        'temperatures': [np.array(b.indoor_dry_bulb_temperature) for b in env.buildings],
        'setpoints': [np.array(b.indoor_dry_bulb_temperature_heating_set_point) for b in env.buildings],
    }

# --- Sanity Checks ---

def run_sanity_checks(results):
    """Run all sanity checks, return list of (check_name, passed, message)."""
    checks = []

    baseline = results['Baseline']
    baseline_energy = baseline['consumption'].sum()

    for name, data in results.items():
        # Energy conservation
        energy_ratio = data['consumption'].sum() / baseline_energy
        passed = 0.85 <= energy_ratio <= 1.15
        checks.append((f"{name}: Energy ratio", passed, f"{energy_ratio:.2f}"))

        # Temperature bounds
        for i, temps in enumerate(data['temperatures']):
            min_t, max_t = temps.min(), temps.max()
            passed = min_t >= 10 and max_t <= 30
            if not passed:
                checks.append((f"{name} Bldg {i}: Temp bounds", False, f"[{min_t:.1f}, {max_t:.1f}]°C"))

        # No NaN values
        has_nan = np.isnan(data['consumption']).any()
        checks.append((f"{name}: No NaN", not has_nan, ""))

    return checks

# --- Charts ---

def plot_daily_profile(results, output_path):
    """Average hourly load by controller."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in results.items():
        hourly = data['consumption'].reshape(-1, 24)
        avg = hourly.mean(axis=0)
        ax.plot(range(24), avg, label=name, linewidth=2)

    ax.axvspan(17, 20, alpha=0.2, color='red', label='Peak Hours')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average District Load (kW)')
    ax.set_title('Daily Load Profile by Controller')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_temperature_trajectory(results, building_idx, hours, output_path):
    """Temperature over time for one building."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, data in results.items():
        temps = data['temperatures'][building_idx][:hours]
        ax.plot(temps, label=name, linewidth=1.5)

    setpoint = results['Baseline']['setpoints'][building_idx][:hours]
    ax.plot(setpoint, 'k--', label='Setpoint', linewidth=1)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Indoor Temperature (°C)')
    ax.set_title(f'Temperature Trajectory - Building {building_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# --- Main ---

def main():
    controllers = {
        'Baseline': lambda e: BaselineAgent(e),
        'Constant_0.5': lambda e: ConstantRBC(e, 0.5),
        'Moderate': lambda e: PeakShavingRBC(e, 'moderate'),
        'Aggressive': lambda e: PeakShavingRBC(e, 'aggressive'),
    }

    print("Running simulations...")
    results = {}
    for name, ctrl_fn in controllers.items():
        print(f"  {name}...")
        results[name] = run_simulation(ctrl_fn)

    # Summary table
    print("\n=== RESULTS ===\n")
    rows = []
    for name, data in results.items():
        kpis = data['kpis']
        district = kpis[kpis['name'] == 'District']

        row = {
            'Controller': name,
            'PAR': calculate_par(data['consumption']),
            'Daily_PAR': calculate_daily_par(data['consumption']),
            'Daily_Peak': district[district['cost_function'] == 'daily_peak_average']['value'].values[0],
            'Total_Energy': district[district['cost_function'] == 'electricity_consumption_total']['value'].values[0],
            'Discomfort_%': district[district['cost_function'] == 'discomfort_proportion']['value'].values[0] * 100,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv('outputs/dr_summary.csv', index=False)

    # Sanity checks
    print("\n=== SANITY CHECKS ===\n")
    checks = run_sanity_checks(results)
    for check, passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check} {msg}")

    # Charts
    print("\nGenerating charts...")
    plot_daily_profile(results, 'outputs/daily_load_profile.png')
    plot_temperature_trajectory(results, 0, 72, 'outputs/temperature_trajectory.png')

    print("\nDone! Outputs in outputs/")

if __name__ == '__main__':
    main()
```

---

## Expected Results

| Metric | Expected Change | Notes |
|--------|-----------------|-------|
| Daily Peak | -10% to -25% | Primary benefit |
| PAR | -5% to -15% | Depends on baseline shape |
| Total Energy | +0% to +10% | Pre-heating overhead |
| Cost | -5% to -15% | If TOU pricing aligned |
| Discomfort | +5% to +15% | Acceptable tradeoff |

---

## Success Criteria

1. **PAR Reduction**: At least 10% reduction in daily peak with moderate strategy
2. **Energy Neutrality**: Total energy within 15% of baseline
3. **Comfort Acceptable**: Discomfort proportion < 20%, max delta < 5°C
4. **Sanity Checks Pass**: No critical failures
5. **Charts Make Sense**: Load shifts visibly from peak to pre-peak hours

---

## Next Steps After Experiment

1. **Price-responsive control**: Use `electricity_pricing` observation instead of fixed schedule
2. **Temperature-aware control**: Adjust action based on current `indoor_dry_bulb_temperature_heating_delta`
3. **MPC integration**: Use thermal predictions to optimize over rolling horizon
4. **Transfer to Ecobee data**: Apply learnings to real thermostat data from your project
