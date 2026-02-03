# Plan: Rule-Based DR Baseline in CityLearn

## Objective
Implement a rule-based demand response (RBC) controller as a baseline for comparison with the ML+MPC approach, and test it in CityLearn.

## Background
Per the proposal, current utility DR programs use uniform setpoint adjustments (e.g., raise cooling setpoint by 4°F during peak hours). This baseline will replicate that behavior to establish performance benchmarks.

## Scope Decisions
- **Seasons**: Test across full year (both heating and cooling scenarios)
- **Building Config**: Use CityLearn's built-in residential schemas to start; customize later if needed

---

## Implementation Steps

### 1. Environment Setup
Create `requirements.txt` and set up CityLearn:
```
citylearn>=2.5.0
numpy
pandas
matplotlib
```

### 2. Rule-Based Controller Implementation

Create `src/baselines/rule_based_controller.py`:

**Strategy 1: Time-of-Use RBC (Primary Baseline)**
- Peak hours (5-8 PM): Reduce HVAC power to 30% of nominal
- Pre-peak (3-5 PM): Pre-condition at 80% power
- Off-peak: Normal operation at 50% power

**Strategy 2: Price-Responsive RBC**
- High price (>$0.15/kWh): Reduce to 20% power
- Medium price: Normal 50% power
- Low price (<$0.08/kWh): Pre-condition at 80% power

**Strategy 3: Simple Setback RBC**
- During DR event window: Apply fixed setback (e.g., +4°F cooling, -4°F heating)
- Outside DR window: Maintain normal setpoint

### 3. Evaluation Harness

Create `src/evaluation/run_simulation.py`:
- Load CityLearn environment with residential building schema
- Run simulation for full episode (1 year or specified period)
- Collect KPIs: discomfort, peak demand, load factor, cost

Create `src/evaluation/metrics.py`:
- Peak-to-Average Ratio (PAR)
- Total electricity cost under TOU pricing
- Comfort violations (degree-hours outside bounds)
- Peak demand reduction vs no-control baseline

### 4. Baseline Comparison

Implement three baselines for comparison:
1. **No Control**: HVAC runs to maintain setpoint (CityLearn default)
2. **Time-of-Use RBC**: Peak-hour power reduction
3. **Price-Responsive RBC**: Dynamic response to price signals

---

## File Structure

```
thermal/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── rule_based_controller.py   # RBC implementations
│   │   └── no_control.py              # Baseline (do nothing)
│   └── evaluation/
│       ├── __init__.py
│       ├── run_simulation.py          # Main simulation runner
│       └── metrics.py                 # Custom KPI calculations
├── configs/
│   └── residential_schema.json        # CityLearn building config
├── scripts/
│   └── run_baseline_comparison.py     # CLI to run all baselines
└── results/
    └── (output CSVs and plots)
```

---

## Key CityLearn Integration Points

**Controller Interface** (inherit from `citylearn.agents.base.Agent`):
```python
class RuleBasedDRController(Agent):
    def predict(self, observations, deterministic=True):
        # Extract: hour, indoor_temp, setpoint, price
        # Apply rule-based logic
        # Return action list
```

**Key Observations to Use**:
- Hour of day (index varies by schema)
- Indoor temperature
- Cooling/heating setpoint
- Electricity price (current + forecasts)

**Action Format**:
- Single float per device: 0.0 to 1.0 (fraction of nominal power)
- Negative for cooling devices

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| PAR | Peak-to-Average Ratio of HVAC load | Lower is better |
| Cost | Total electricity cost (TOU pricing) | Lower is better |
| Discomfort | Degree-hours outside comfort band | Lower is better |
| Peak Reduction | % reduction vs no-control baseline | Higher is better |

---

## Verification

1. Run `scripts/run_baseline_comparison.py`
2. Verify outputs in `results/`:
   - `baseline_comparison.csv` with KPIs for each strategy
   - `load_profile_summer.png` and `load_profile_winter.png` showing seasonal demand curves
   - `comfort_violations.png` showing temperature deviations
3. Confirm PAR reduction for RBC vs no-control baseline in both seasons
4. Confirm discomfort stays within acceptable bounds
5. Compare heating vs cooling season performance

---

## Dependencies

- Python 3.9+
- CityLearn 2.5.0+
- numpy, pandas, matplotlib

---

## Notes

- CityLearn includes built-in RBC examples (`HourRBC`, `BasicRBC`) that can serve as templates
- May need to create/modify building schema JSON for residential single-family homes
- Start with CityLearn's example schemas before creating custom ones
