# Data Analysis: Temperature Change Distribution

## Question: Does 30 minutes make sense as a prediction horizon?

### Temperature Change by Horizon and Mode

| Horizon | Mode | % of Data | Mean ΔT | Std | P5/P95 | % < 0.5°F |
|---------|------|-----------|---------|-----|--------|-----------|
| **15 min** | Heating | 19.8% | +0.56°F | 1.85°F | -2/+4°F | 36% |
| | Cooling | 7.1% | -0.81°F | 2.49°F | -5/+3°F | 18% |
| | Passive | 73.1% | -0.07°F | 1.91°F | -3/+3°F | 29% |
| **30 min** | Heating | 19.7% | +0.74°F | 2.27°F | -3/+5°F | 30% |
| | Cooling | 7.1% | -1.18°F | 3.06°F | -6/+4°F | 14% |
| | Passive | 73.2% | -0.08°F | 2.45°F | -4/+4°F | 24% |
| **1 hr** | Heating | 19.6% | +0.91°F | 2.76°F | -3/+6°F | 24% |
| | Cooling | 7.1% | -1.66°F | 3.58°F | -8/+4°F | 11% |
| | Passive | 73.3% | -0.08°F | 2.93°F | -5/+5°F | 19% |
| **2 hr** | Heating | 19.3% | +1.08°F | 3.32°F | -4/+7°F | 19% |
| | Cooling | 7.2% | -2.16°F | 3.99°F | -9/+4°F | 9% |
| | Passive | 73.5% | -0.08°F | 3.41°F | -6/+6°F | 16% |

### Key Insights

1. **Mode matters a lot:**
   - **Heating:** Clear upward signal (+0.74°F avg at 30 min)
   - **Cooling:** Clear downward signal (-1.18°F avg at 30 min)
   - **Passive:** Essentially flat (-0.08°F avg)

2. **Passive mode dominates (73%):**
   - This is why persistence baseline works well overall
   - HVAC is doing its job of maintaining temperature in the deadband
   - But in heating/cooling modes, there's real signal to exploit

3. **Cooling has biggest changes:**
   - Std 3.1°F at 30 min (vs 2.3°F for heating)
   - AC units are powerful and create rapid temperature drops

4. **Longer horizons increase signal but also noise:**
   - Mean change grows: +0.56 → +0.74 → +0.91 → +1.08°F (heating)
   - Std also grows: 1.85 → 2.27 → 2.76 → 3.32°F
   - Diminishing returns on signal-to-noise ratio

### Recommendation

**30 minutes is a reasonable horizon because:**

1. **Enough signal in active modes** - Mean changes of +0.74°F (heating) and -1.18°F (cooling) are large enough to distinguish from noise

2. **Matches MPC use case** - Typical MPC receding horizon is 15-60 minutes for HVAC control

3. **Practical for DR events** - Demand response events typically provide 15-30 min advance notice

4. **Not so long that uncertainty dominates** - At 2 hours, std exceeds 3°F making accurate prediction difficult

### Evaluation Strategy

Since passive mode dominates and has low signal, we should:

1. **Report metrics stratified by mode** - Overall MAE hides where models actually help
2. **Weight evaluation toward active modes** - Or report separate MAE for heating/cooling/passive
3. **Focus model improvements on active modes** - This is where RNN should beat baselines

### Mode Distribution

```
Passive:  ████████████████████████████████████ 73%
Heating:  ████████ 20%
Cooling:  ███ 7%
```

Most of the time, temperature stays stable (HVAC maintaining setpoint). The interesting prediction challenge is in the 27% of time when HVAC is actively heating or cooling.

---

## Setpoint Deviation Analysis

### Mode Distribution Varies Hugely Across Homes

| Mode | Mean | Median | Min | Max | P10 | P90 |
|------|------|--------|-----|-----|-----|-----|
| Passive | 73.5% | 76.2% | 10.7% | 99.6% | 49.7% | 91.1% |
| Heating | 19.8% | 17.3% | 0.1% | 81.8% | 5.2% | 37.6% |
| Cooling | 6.8% | 3.8% | 0.0% | 67.0% | 0.8% | 16.4% |

Some homes are almost always passive (99.6%), others spend 82% of time heating. This heterogeneity explains why global models struggle.

---

### Episode Duration

How long do heating/cooling episodes last?

#### Heating Episodes (2.3M total)

| Metric | Value |
|--------|-------|
| Median | 25 min |
| Mean | 42 min |
| P75 | 40 min |
| P90 | 70 min |
| P99 | 330 min |

**Duration distribution:**
```
≤5 min:    7.1%
5-10 min: 10.7%
10-15 min: 15.3%
15-30 min: 35.0%  ← Most common
30-60 min: 20.0%
1-2 hr:    8.1%
2-4 hr:    2.4%
>4 hr:     1.4%
```

**68% of heating episodes are ≤30 minutes**

#### Cooling Episodes (500K total)

| Metric | Value |
|--------|-------|
| Median | 15 min |
| Mean | 67 min |
| P75 | 40 min |
| P90 | 120 min |
| P99 | 830 min |

**Duration distribution:**
```
≤5 min:   17.9%  ← More short episodes
5-10 min: 14.1%
10-15 min: 13.9%
15-30 min: 25.5%
30-60 min: 13.9%
1-2 hr:    6.7%
2-4 hr:    3.7%
>4 hr:     4.4%
```

**71% of cooling episodes are ≤30 minutes**

---

### Gap From Setpoint

How far outside the deadband do homes get?

#### Heating (19.4M samples where indoor < heat_setpoint)

| Metric | Value |
|--------|-------|
| Median | 1.0°F |
| Mean | 1.67°F |
| P90 | 3.0°F |
| P99 | 9.0°F |

**Gap distribution:**
```
≤0.5°F:   0.3%
0.5-1°F: 75.6%  ← Vast majority barely below setpoint
1-2°F:   10.1%
2-3°F:    4.6%
3-5°F:    4.9%
5-10°F:   3.9%
>10°F:    0.6%
```

**76% of heating samples are ≤1°F from setpoint**

#### Cooling (6.7M samples where indoor > cool_setpoint)

| Metric | Value |
|--------|-------|
| Median | 2.0°F |
| Mean | 2.92°F |
| P90 | 6.0°F |
| P99 | 13.0°F |

**Gap distribution:**
```
≤0.5°F:   0.1%
0.5-1°F: 41.3%
1-2°F:   19.1%
2-3°F:   12.0%
3-5°F:   13.6%
5-10°F:  11.4%
>10°F:    2.5%
```

Cooling has **larger gaps** than heating (median 2°F vs 1°F). AC systems may be less responsive, or homes experience larger heat gains than losses.

---

### Implications for Prediction

1. **Most episodes are short (≤30 min)**
   - Many heating/cooling episodes end within our prediction horizon
   - At episode end, temp is at setpoint → "no change" is correct
   - This favors persistence baseline

2. **Gaps are tiny**
   - 76% of heating is within 1°F of setpoint
   - HVAC is barely "catching up" - it's maintaining well
   - Small gap → small temperature change → persistence wins

3. **Episode transitions are hard to predict**
   - When will HVAC turn off? Depends on when temp reaches setpoint
   - Linear models can't capture this threshold behavior

4. **Cooling has more signal**
   - Larger gaps (median 2°F vs 1°F)
   - More variance in episode duration
   - May be easier to beat persistence in cooling mode

5. **Home heterogeneity is extreme**
   - Some homes: 82% heating, 0% cooling (cold climate)
   - Some homes: 67% cooling, 0% heating (hot climate)
   - Global models must handle this diversity

---

## Setpoint Change Analysis

For MPC-based precooling/preheating, we need to predict response to interventions, not passive forecasting. How much setpoint change data do we have?

### Setpoint Changes Are Abundant

| Metric | Heat Setpoint | Cool Setpoint | Total |
|--------|---------------|---------------|-------|
| Total changes (≥1°F) | 1,740,731 | 1,682,142 | **3,422,873** |
| Per home (median) | 1,780 | 1,677 | 3,472 |
| Per home (P10) | 441 | 482 | 1,025 |
| Per home (P90) | 3,003 | 2,916 | 5,803 |

**90% of homes have >1,000 setpoint changes** - plenty of data for incident-based modeling.

Only 2 homes (0.2%) have zero setpoint changes.

### Homes by Change Frequency

```
0 changes:        2 homes (0.2%)
1-10 changes:     1 homes (0.1%)
11-50 changes:    5 homes (0.5%)
51-100 changes:   2 homes (0.2%)
101-500 changes: 38 homes (3.9%)
501-1000:        48 homes (4.9%)
>1000 changes:  875 homes (90.1%)  ← Most homes
```

### Change Magnitudes

| Magnitude | Heat Setpoint | Cool Setpoint |
|-----------|---------------|---------------|
| 1-2°F | 38.6% | 40.1% |
| 2-3°F | 22.9% | 23.9% |
| 3-5°F | 24.5% | 23.1% |
| 5-10°F | 12.2% | 11.0% |
| >10°F | 1.7% | 1.9% |

Changes are roughly 50/50 increases vs decreases (thermostat schedules: up in morning, down at night).

---

## Active vs Passive Thermal Modification

**Key distinction**: Is HVAC fighting the thermal gradient, or does outdoor temperature help?

- **Active**: HVAC working against thermal gradient (outdoor temp makes it harder)
- **Passive**: Thermal gradient helps (outdoor temp assists HVAC goal)

### Results

| Mode | Total | HVAC Fighting Gradient | Gradient Helping |
|------|-------|------------------------|------------------|
| **Heating** | 19.7% of samples | **92.2% active** | 7.8% passive |
| **Cooling** | 6.8% of samples | **35.7% active** | 64.3% passive |
| **Deadband** | 73.5% of samples | - | - |

### Key Insight: Heating vs Cooling Dynamics

**Heating is almost always active (92%)**
- When indoor < heat_setpoint, outdoor is usually even colder
- HVAC is clearly doing the work
- Clean signal for modeling

**Cooling is mostly passive (64%)**
- When indoor > cool_setpoint, outdoor is often *cooler* than inside
- Common scenario: heat built up during day, evening outdoor has cooled
- Home would drift toward setpoint even without AC
- Confounded signal - hard to isolate AC effect

### Summary of Thermal Dynamics

```
HVAC actively fighting gradient:  20.6% of all samples
Thermal gradient helping HVAC:     5.9% of all samples
Passive (within deadband):        73.5% of all samples
```

---

## Implications for Incident-Based Modeling

### Why Incident-Based Makes Sense

Current approach (predict temp at arbitrary times) is dominated by:
- 73% passive mode where nothing happens
- 76% of heating within 1°F of setpoint (tiny signal)

For MPC, we need to answer: **"If I change the setpoint, what happens?"**

### Recommended Incident Types

1. **Active Heating Events**
   - Trigger: Setpoint raised when outdoor < indoor
   - Predict: Time to reach setpoint, temperature trajectory
   - Clean signal: 92% of heating is this case

2. **Active Cooling Events**
   - Trigger: Setpoint lowered when outdoor > indoor
   - Predict: Time to reach setpoint, temperature trajectory
   - Only 36% of cooling - be selective

3. **Drift Events**
   - Trigger: Transition to passive mode (HVAC turns off)
   - Predict: Drift rate toward outdoor (°F/hour)
   - Critical for "how long until comfort violation?"

### Data Availability

| Incident Type | Estimated Count | Notes |
|---------------|-----------------|-------|
| Heat setpoint increases | ~870K | 50% of 1.74M heat changes |
| Cool setpoint decreases | ~840K | 50% of 1.68M cool changes |
| Active heating starts | ~2.1M | Based on episode count × 92% |
| Active cooling starts | ~180K | Based on episode count × 36% |

Plenty of data for incident-based modeling, especially for heating.
