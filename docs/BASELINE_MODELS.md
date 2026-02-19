# Baseline Models - Detailed Explanation

## The Physics of Indoor Temperature Change

Temperature change over 30 minutes is driven by:

```
ΔT_indoor = Heat_in - Heat_out

Heat_out = k₁ × (indoor - outdoor)     # Conduction/convection loss
Heat_in  = HVAC_output + Solar + Internal_gains
```

### What the thermostat knows:

| Observable | What it tells us |
|------------|------------------|
| `(outdoor - indoor)` | Thermal driving force - how fast heat flows through walls |
| `indoor < heat_sp` | Heating is ON → temp rising toward setpoint |
| `indoor > cool_sp` | Cooling is ON → temp falling toward setpoint |
| `heat_sp ≤ indoor ≤ cool_sp` | HVAC OFF → temp drifts toward outdoor |
| `(heat_sp - indoor)` when heating | How hard HVAC is working / how long until satisfied |
| `hour` | Solar gain (afternoon), scheduled setpoint changes |

### Key insight: Mode-specific behavior

| Mode | What happens | Predict drift toward |
|------|--------------|---------------------|
| Heating | Furnace ON | `heat_sp` |
| Cooling | AC ON | `cool_sp` |
| Passive | HVAC OFF | `outdoor` (mediated by insulation) |

---

## B1: Persistence

**Idea:** Temperature doesn't change much in 30 minutes. Just predict it stays the same.

**Formula:**
```
y_pred = indoor_temp[t]
```

**Features:** None (just current temperature)

**Why it works:** Indoor temperature has high inertia. Over 30 minutes, most homes change < 0.5°F. This is the baseline everything must beat.

**When it fails:**
- Right after setpoint change (HVAC kicks on hard)
- Extreme outdoor temps (fast drift)

---

## B2: Thermal Drift

**Idea:** Heat flows from hot to cold. When HVAC is off, indoor temp drifts toward outdoor temp at a rate proportional to the temperature difference.

**Formula:**
```
thermal_drive = outdoor - indoor
y_pred = indoor + k × thermal_drive
```

**Features:**
- `indoor_temp`
- `outdoor_temp`
- `(outdoor - indoor)` — the thermal driving force

**Fit:** Learn `k` from training data (linear regression with one feature)

**Physics:** This is Newton's law of cooling. If outdoor is 40°F and indoor is 70°F, heat flows out. The bigger the gap, the faster the heat loss.

**When it fails:**
- Ignores HVAC completely
- If heating is ON, temp rises even when outdoor < indoor

---

## B3: Mode-Aware Target

**Idea:** Detect which mode the HVAC is in, then predict temperature drifts toward the appropriate target.

**Mode Detection:**
```
if indoor < heat_setpoint:
    mode = HEATING
    target = heat_setpoint      # HVAC pushing toward this

elif indoor > cool_setpoint:
    mode = COOLING
    target = cool_setpoint      # HVAC pushing toward this

else:
    mode = PASSIVE
    target = outdoor_temp       # No HVAC, drift toward outside
```

**Formula:**
```
y_pred = indoor + k[mode] × (target - indoor)
```

**Features:**
- `indoor_temp`
- `outdoor_temp`
- `heat_setpoint`
- `cool_setpoint`

**Fit:** Learn 3 drift rates: `k_heat`, `k_cool`, `k_passive`

**Why it's better than B2:**
- Knows that when heating is ON, temp moves toward setpoint (not outdoor)
- Different drift rates for different modes

**When it fails:**
- Fixed drift rate per mode (doesn't account for HOW FAR from setpoint)
- Doesn't know how powerful the HVAC system is

---

## B4: LinReg + Thermal

**Idea:** Linear regression with physics-informed features. Let the model learn the relationships.

**Features:**
```
X = [
    indoor_temp,
    outdoor_temp,
    (outdoor - indoor),     # Thermal driving force
    heat_setpoint,
    cool_setpoint,
]
```

**Formula:**
```
y_pred = w0 + w1×indoor + w2×outdoor + w3×(out-in) + w4×heat_sp + w5×cool_sp
```

**Why include `(outdoor - indoor)` separately?**

Even though it's derivable from `outdoor` and `indoor`, including it explicitly helps the model. Linear regression finds:
- `w2` captures outdoor's direct effect
- `w3` captures the thermal drive effect

The model might learn something like: "temperature stays near current (`w1` ≈ 1), but drifts slightly toward outdoor (`w3` small positive)."

**When it fails:**
- Doesn't explicitly know about HVAC modes
- Same coefficients whether heating or cooling

---

## B5: LinReg + Mode

**Idea:** Add binary indicators for HVAC mode so the model can learn different behavior for heating vs cooling vs passive.

**Features:**
```
is_heating = 1 if indoor < heat_setpoint else 0
is_cooling = 1 if indoor > cool_setpoint else 0
# (passive is when both are 0)

X = [
    indoor_temp,
    outdoor_temp,
    (outdoor - indoor),
    heat_setpoint,
    cool_setpoint,
    is_heating,             # NEW
    is_cooling,             # NEW
]
```

**Formula:**
```
y_pred = w0 + w1×indoor + ... + w6×is_heating + w7×is_cooling
```

**What the model learns:**
- `w6` (is_heating coefficient): How much to adjust prediction when heating is ON
- `w7` (is_cooling coefficient): How much to adjust prediction when cooling is ON

If `w6 > 0`: When heating, predict higher temp (makes sense!)
If `w7 < 0`: When cooling, predict lower temp (makes sense!)

**Why it's better than B4:**
- Can have different biases for each mode
- Captures "heating pushes temp up, cooling pushes temp down"

---

## B6: LinReg + Gap

**Idea:** Add how far the current temperature is from the target setpoint. This tells us how hard the HVAC is working.

**Features:**
```
if is_heating:
    gap_to_target = heat_setpoint - indoor    # Positive = need to heat more
elif is_cooling:
    gap_to_target = indoor - cool_setpoint    # Positive = need to cool more
else:
    gap_to_target = 0                         # HVAC off

X = [
    indoor_temp,
    outdoor_temp,
    (outdoor - indoor),
    heat_setpoint,
    cool_setpoint,
    is_heating,
    is_cooling,
    gap_to_target,          # NEW
]
```

**Why this matters:**

Consider two heating scenarios:
- **Scenario A:** indoor = 67°F, heat_sp = 68°F → gap = 1°F (almost satisfied, HVAC about to turn off)
- **Scenario B:** indoor = 62°F, heat_sp = 68°F → gap = 6°F (far from target, HVAC running hard)

In Scenario B, we expect MORE temperature rise because HVAC runs the full 30 minutes.

**What the model learns:**
- Larger gap → more temperature change in the direction of setpoint

---

## B7: LinReg + Time

**Idea:** Add time-of-day features to capture solar gain and schedule patterns.

**Features:**
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

X = [
    indoor_temp,
    outdoor_temp,
    (outdoor - indoor),
    heat_setpoint,
    cool_setpoint,
    is_heating,
    is_cooling,
    gap_to_target,
    hour_sin,               # NEW
    hour_cos,               # NEW
]
```

**Why cyclical encoding?**
- Hour 23 and hour 0 are close (both nighttime), but numerically 23 vs 0 looks far apart
- sin/cos encoding puts them close in feature space

**What the model learns:**
- Afternoon (positive hour_sin) → solar gain → slight warming bias
- Night (negative hour_sin) → no solar → slight cooling bias

---

## B8: LinReg Full

**Idea:** Combine all features from B4-B7.

**Features:**
```
X = [
    indoor_temp,
    outdoor_temp,
    (outdoor - indoor),
    heat_setpoint,
    cool_setpoint,
    is_heating,
    is_cooling,
    gap_to_target,
    hour_sin,
    hour_cos,
]
```

**Total:** 10 features

**This is the "kitchen sink" linear model.** It has all the physics insight and mode awareness. If linear regression can solve this problem, this model should do it.

---

## B9: Per-Mode LinReg

**Idea:** Instead of one model with mode indicators, train 3 completely separate models.

**Implementation:**
```python
# Split data by mode
heating_samples = data[data.indoor < data.heat_sp]
cooling_samples = data[data.indoor > data.cool_sp]
passive_samples = data[(data.indoor >= data.heat_sp) & (data.indoor <= data.cool_sp)]

# Train separate models
model_heat = LinearRegression().fit(heating_samples)
model_cool = LinearRegression().fit(cooling_samples)
model_passive = LinearRegression().fit(passive_samples)

# At prediction time, route to appropriate model
if indoor < heat_sp:
    y_pred = model_heat.predict(...)
elif indoor > cool_sp:
    y_pred = model_cool.predict(...)
else:
    y_pred = model_passive.predict(...)
```

**Features per model:**
```
Heating model:   indoor, outdoor, (out-in), heat_sp, (heat_sp - indoor), hour_sin, hour_cos
Cooling model:   indoor, outdoor, (out-in), cool_sp, (indoor - cool_sp), hour_sin, hour_cos
Passive model:   indoor, outdoor, (out-in), hour_sin, hour_cos
```

**Why this might be better:**
- Each mode has completely different dynamics
- Heating model doesn't waste capacity learning cooling behavior
- Can have different feature importances per mode

**Why it might not help:**
- Less training data per model (split 3 ways)
- Mode boundaries might be noisy (indoor ≈ setpoint)

---

## Summary Table

| Model | Key Idea | # Features |
|-------|----------|------------|
| B1 | No change | 0 |
| B2 | Drift toward outdoor | 1 |
| B3 | Drift toward mode-specific target | 3 params |
| B4 | Linear with thermal drive | 5 |
| B5 | + Mode indicators | 7 |
| B6 | + Gap to setpoint | 8 |
| B7 | + Time of day | 10 |
| B8 | All features combined | 10 |
| B9 | Separate models per mode | 3 × ~6 |
