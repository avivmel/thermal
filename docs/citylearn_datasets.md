# CityLearn Datasets Reference

Research notes on available CityLearn datasets for demand response baseline experiments.

## Overview

CityLearn is a simulation environment for grid-interactive buildings. Most datasets include batteries, PV, and storage systems. For simple HVAC-only control experiments, dataset selection is critical.

## Available Datasets

```
baeda_3dem
ca_alameda_county_neighborhood
citylearn_challenge_2020_climate_zone_1
citylearn_challenge_2020_climate_zone_2
citylearn_challenge_2020_climate_zone_3
citylearn_challenge_2020_climate_zone_4
citylearn_challenge_2021
citylearn_challenge_2022_phase_1
citylearn_challenge_2022_phase_2
citylearn_challenge_2022_phase_3
citylearn_challenge_2022_phase_all
citylearn_challenge_2022_phase_all_plus_evs
citylearn_challenge_2023_phase_1
citylearn_challenge_2023_phase_2_local_evaluation
citylearn_challenge_2023_phase_2_online_evaluation_1
citylearn_challenge_2023_phase_2_online_evaluation_2
citylearn_challenge_2023_phase_2_online_evaluation_3
citylearn_challenge_2023_phase_3_1
citylearn_challenge_2023_phase_3_2
citylearn_challenge_2023_phase_3_3
citylearn_charging_constraints_demo
quebec_neighborhood_with_demand_response_set_points
quebec_neighborhood_without_demand_response_set_points
tx_travis_county_neighborhood
vt_chittenden_county_neighborhood
```

## Dataset Comparison

| Dataset | Buildings | Duration | Actions | Equipment | Best For |
|---------|-----------|----------|---------|-----------|----------|
| **quebec_neighborhood_with_demand_response_set_points** | 20 | 90 days | `heating_device` | Heat pumps | Simple HVAC control |
| citylearn_challenge_2022_phase_all | 17 | 365 days | `electrical_storage` | PV + Battery + HVAC | Battery optimization |
| citylearn_challenge_2020_climate_zone_* | 9 | 365 days | `cooling_storage`, `dhw_storage`, `electrical_storage` | Full DER stack | Storage research |
| baeda_3dem | 4 | 122 days | `cooling_storage`, `cooling_device` | Mixed | Cooling control |

---

## Recommended: Quebec Heating Dataset

**Dataset**: `quebec_neighborhood_with_demand_response_set_points`

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| Buildings | 20 residential |
| Duration | 2,160 hours (90 days) |
| Season | Winter (starts January) |
| Time Resolution | Hourly |
| Action Type | Direct heating power control |
| Central Agent | Yes (single action array) |

### Equipment per Building

- **Heating Device**: Heat pump (14-47 kW nominal power)
- **Cooling Device**: Yes (unused in winter)
- **Electrical Storage**: Present but not in action space
- **PV**: Present but not in action space
- **DHW**: Yes

**Total heating capacity**: ~616 kW across all 20 buildings

### Action Space

```python
# Central agent mode - single array of 20 values
env.action_space[0].shape  # (20,)
env.central_agent  # True

# Action format: [[val1, val2, ..., val20]]
actions = [[0.5] * 20]  # 50% power to all buildings
env.step(actions)
```

Action values are power fractions (0.0 to 1.0) for each building's heat pump.

### Load Profile (at 50% heating)

```
Hour  0-5:   315-325 kW (overnight)
Hour  6-10:  340-350 kW (morning warmup)
Hour 11-16:  330-345 kW (daytime)
Hour 17-20:  340-350 kW (evening peak)
Hour 21-23:  320-336 kW (evening decline)

Min: 109.5 kW
Max: 422.1 kW
Avg: 304.3 kW
```

### Usage Example

```python
from citylearn.data import DataSet
from citylearn.citylearn import CityLearnEnv

ds = DataSet()
schema = ds.get_schema('quebec_neighborhood_with_demand_response_set_points')
env = CityLearnEnv(schema=schema)

obs, _ = env.reset()

# Central agent: single list with 20 action values
actions = [[0.5] * 20]  # 50% heating for all buildings
obs, rewards, terminated, truncated, info = env.step(actions)
```

---

## CityLearn Challenge 2022 Dataset

**Dataset**: `citylearn_challenge_2022_phase_all`

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| Buildings | 17 commercial/residential |
| Duration | 8,760 hours (365 days) |
| Season | Starts July |
| Time Resolution | Hourly |
| Action Type | Battery charge/discharge |
| Central Agent | No |

### Equipment per Building

- **Cooling Device**: Yes
- **Heating Device**: Yes
- **Electrical Storage**: Yes (primary action target)
- **PV**: Yes (solar generation)

### Action Space

```python
# Decentralized - one action per building
env.action_names[0]  # ['electrical_storage']
env.buildings[0].action_space  # Box(-1.0, 1.0, (1,), float32)

# Action format: [[val]] per building
actions = [[0.5] for _ in env.buildings]
```

- Positive values (0 to 1): Charge battery
- Negative values (-1 to 0): Discharge battery

### Electricity Pricing

```
Off-peak (hours 0-15, 21-23): $0.22/kWh
Peak (hours 16-20):           $0.54/kWh
```

### Why RBC Baseline Failed

The initial spike at hour 1 (136.6 kW) came from **battery charging at simulation start**, not HVAC load:

| Hour | Net Power | Battery Consumption |
|------|-----------|---------------------|
| 1 | 136.6 kW | 85.0 kW |
| 2 | 54.8 kW | 42.5 kW |
| 3 | 40.9 kW | 29.9 kW |

TimeOfUseRBC targeted hours 17-20, but the peak occurred at hour 1 from battery initialization.

---

## Other Notable Datasets

### baeda_3dem

- **4 buildings**, 122 days (summer)
- Actions: `cooling_storage` + `cooling_device`
- Good for cooling-focused experiments
- Mixed action space (different buildings have different actions)

### Climate Zone Datasets (2020)

- **9 buildings**, 365 days
- 4 climate zones available
- Actions: `cooling_storage`, `dhw_storage`, `electrical_storage`
- Full distributed energy resource stack

---

## Loading Any Dataset

```python
from citylearn.data import DataSet
from citylearn.citylearn import CityLearnEnv

ds = DataSet()

# List all available datasets
print(ds.get_dataset_names())

# Load specific dataset
schema = ds.get_schema('dataset_name')
env = CityLearnEnv(schema=schema)

# Check configuration
print(f"Buildings: {len(env.buildings)}")
print(f"Time steps: {env.time_steps}")
print(f"Actions: {env.action_names}")
print(f"Central agent: {env.central_agent}")
```

---

## Recommendations

1. **For simple HVAC DR baselines**: Use `quebec_neighborhood_with_demand_response_set_points`
   - Direct heating control
   - Clear daily load patterns
   - No battery artifacts

2. **For battery/storage research**: Use `citylearn_challenge_2022_phase_all`
   - Full DER stack
   - Real pricing data
   - Competition benchmark

3. **For cooling research**: Use `baeda_3dem`
   - Direct cooling device control
   - Summer season
   - Smaller scale (4 buildings)
