# Setpoint Response Baselines

## Task Definition

Predict indoor temperature 15 minutes ahead during setpoint response episodes.

| Attribute | Value |
|-----------|-------|
| Dataset | `data/setpoint_responses.parquet` |
| Horizon | 15 minutes (3 timesteps) |
| Metric | MAE (°F) |
| Evaluation | Rolling windows within each episode |

### Critical Finding: Temperature Quantization

**The Ecobee sensors report whole degrees Fahrenheit only.** This fundamentally affects prediction:

| Observation | Value |
|-------------|-------|
| 15-min windows with exactly 0°F change | 66% |
| Mean temperature change | +0.09°F |
| Median temperature change | 0°F |

Temperature trajectories are stair-steps, not smooth curves:
```
69→69→69→69→69→70  (sits at 69°F for 25min, then jumps)
77→77→77→77→76→76→76→76→75→75→75→75→74
```

**This explains why persistence wins**: predicting "no change" is literally correct 66% of the time because most 15-min windows don't cross an integer boundary.

### Key Difference from Flat Dataset

The flat dataset samples arbitrary timesteps where ~73% are passive (no change). The setpoint response dataset contains only active episodes where HVAC is driving temperature toward a known target. Despite this:

- **Persistence still wins on MAE** due to integer quantization
- **Per-home models help significantly** when trained on same home
- **RMSE shows more differentiation** than MAE

---

## Window Sampling

For each episode, create overlapping 15-minute prediction windows:

```
Episode timeline (5-min resolution):
t=0    t=1    t=2    t=3    t=4    t=5    t=6    ...
|------|------|------|------|------|------|------|

Window 0: predict temp[t=3] from state at t=0
Window 1: predict temp[t=4] from state at t=1
Window 2: predict temp[t=5] from state at t=2
...
```

Each window provides:
- `current_temp`: Indoor temperature at window start
- `target_setpoint`: The setpoint being approached
- `start_temp`: Temperature when episode began
- `timestep_idx`: Position within episode (0-indexed)
- `y_true`: Actual temperature 15 min later

---

## Baselines

### B1: Persistence

**Idea**: Temperature doesn't change much in 15 minutes.

```
y_pred = current_temp
```

**Expected behavior**: Strong baseline in flat dataset (0.43°F MAE), but should be weaker here since all episodes are actively changing temperature.

---

### B2: Target Reached

**Idea**: HVAC reaches the setpoint within 15 minutes.

```
y_pred = target_setpoint
```

**Expected behavior**: Good when:
- Current temp is already close to target
- Late in episode (about to reach target)

Poor when:
- Large gap remains
- Early in episode

---

### B3: Linear Interpolation

**Idea**: Extrapolate the observed rate of temperature change.

```
elapsed_min = timestep_idx * 5
temp_change_so_far = current_temp - start_temp

if elapsed_min > 0:
    rate = temp_change_so_far / elapsed_min  # °F/min
else:
    rate = 0

y_pred = current_temp + rate * 15
```

**Expected behavior**: Works when heating/cooling rate is steady. Fails when:
- Rate slows near setpoint (overpredicts change)
- Early in episode with no history (rate = 0)

---

### B4: Gap Fraction

**Idea**: Predict that a learned fraction of the remaining gap is closed in 15 minutes.

```
current_gap = target_setpoint - current_temp  # signed
y_pred = current_temp + k * current_gap
```

**Parameter**: Learn `k` from training data (minimize MAE).

**Why this makes sense**:
- Large gap → predict large change
- Small gap → predict small change (natural "soft landing")
- Single parameter captures average system response

**Expected behavior**: Should be competitive because it naturally handles diminishing returns near setpoint.

---

### B5: Clipped Gap Fraction

**Idea**: Same as B4, but prevent overshoot.

```
current_gap = target_setpoint - current_temp
y_pred = current_temp + k * current_gap

# Clip to not overshoot target
if change_type == "heat_increase":
    y_pred = min(y_pred, target_setpoint)
else:  # cool_decrease
    y_pred = max(y_pred, target_setpoint)
```

**Expected behavior**: Slightly better than B4 when predictions would overshoot, especially for small gaps.

---

### B6: Mode-Specific Gap Fraction

**Idea**: Learn separate gap fractions for heating vs cooling.

```
if change_type == "heat_increase":
    y_pred = current_temp + k_heat * current_gap
else:
    y_pred = current_temp + k_cool * current_gap
```

**Parameters**: `k_heat`, `k_cool` learned separately.

**Rationale**: Heating and cooling have different dynamics:
- Furnaces heat faster than AC cools
- Different thermal mass effects

---

### B7: Time-Aware Gap Fraction

**Idea**: Gap fraction depends on episode progress.

```
# Discretize episode stage
if timestep_idx <= 2:      # First 10 min
    stage = "early"
elif timestep_idx <= 5:    # 10-25 min
    stage = "mid"
else:                      # >25 min
    stage = "late"

y_pred = current_temp + k[stage] * current_gap
```

**Parameters**: `k_early`, `k_mid`, `k_late` learned separately.

**Rationale**:
- Early: System ramping up, may close gap faster
- Late: Approaching setpoint, rate slows down

---

## Expected Results

| Baseline | Hypothesis |
|----------|------------|
| B1: Persistence | Weak (unlike flat dataset, temp is changing) |
| B2: Target Reached | Weak overall, strong for small gaps |
| B3: Linear Interp | Moderate, fails near setpoint |
| B4: Gap Fraction | Strong (captures diminishing returns) |
| B5: Clipped Gap | Slightly better than B4 |
| B6: Mode-Specific | Better than B4 if heat/cool differ |
| B7: Time-Aware | Best if dynamics vary by episode stage |

**Prediction**: B4-B7 (gap-based methods) will outperform B1-B3 because they leverage knowledge of the target and naturally handle the approach dynamics.

---

## Stratified Evaluation

Report metrics broken down by:

1. **Change type**: `heat_increase` vs `cool_decrease`
2. **Current gap size**: How far from target at prediction time
3. **Episode stage**: Early/mid/late in episode

This reveals where each baseline succeeds or fails.

---

## Script

```bash
# Cross-home evaluation (train/test on different homes)
python scripts/run_setpoint_baselines.py

# Within-home evaluation (train/test on different episodes from same homes)
python scripts/run_setpoint_baselines.py --within-home
```

---

## Experimental Results

### Within-Home Split (Per-Home Models Can Learn)

| Model | MAE | RMSE | P95 | Bias |
|-------|-----|------|-----|------|
| B1: Persistence | **0.433** | 0.801 | 2.000 | -0.092 |
| B2: Target Reached | 1.948 | 2.595 | 6.000 | +0.118 |
| B3: Linear Interp | 0.522 | 0.875 | 2.000 | -0.026 |
| B8: Thermal Drive | 0.499 | 0.771 | 1.768 | -0.004 |
| B10: Per-Home Gap | 0.484 | 0.667 | 1.410 | -0.035 |
| **B11: Per-Home Thermal** | 0.475 | **0.661** | **1.367** | -0.005 |

### By Episode Stage

| Model | Early (0-10min) | Mid (10-30min) | Late (>30min) |
|-------|-----------------|----------------|---------------|
| B1: Persistence | 0.891 | 0.611 | **0.274** |
| B11: Per-Home Thermal | **0.631** | **0.519** | 0.423 |

**Key insight**: Per-home models excel early/mid episode (-29% and -15% vs persistence), but persistence dominates late stage when temp is stable.

### By Gap Size

| Model | 0-1°F | 1-2°F | 2-3°F | 3-5°F | >5°F |
|-------|-------|-------|-------|-------|------|
| B1: Persistence | 0.251 | **0.183** | 0.576 | 0.687 | **0.566** |
| B11: Per-Home | **0.172** | 0.326 | **0.511** | **0.607** | 0.735 |

### Learned Parameters

Gap fraction `k` (fraction of gap closed in 15 min):
- Global: k = 0.129
- Heat increase: k = 0.163
- Cool decrease: k = 0.093
- Early stage: k = 0.347
- Mid stage: k = 0.253
- Late stage: k = 0.065

---

## Key Takeaways

1. **Persistence wins on MAE** due to 66% of windows having exactly 0°F change (integer quantization)

2. **Per-home models win on RMSE** (-17% vs persistence) and excel in early/mid episode stages

3. **Cross-home vs within-home matters**: Per-home models only help when you have training data from the same home

4. **Consider classification**: Given integer quantization, predicting {-1, 0, +1}°F may be more appropriate than regression
