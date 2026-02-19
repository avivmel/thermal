# Baseline Results

## Configuration

- **Horizon:** 6 timesteps (30 minutes)
- **Sample rate:** Every 72 timesteps (6 hours)
- **Train samples:** 954,995 (679 homes)
- **Test samples:** 279,682 (195 homes)

## Mode Distribution

| Mode | Test Samples | % |
|------|--------------|---|
| Passive | 203,269 | 72.7% |
| Heating | 55,392 | 19.8% |
| Cooling | 21,021 | 7.5% |

---

## Overall Results

| Model | MAE | RMSE | Max | P95 | Bias | R² |
|-------|-----|------|-----|-----|------|-----|
| B1: Persistence | **0.434** | 0.772 | 11.00 | 1.000 | -0.001 | 0.966 |
| B2: Thermal Drift | 0.452 | 0.771 | 10.89 | 1.085 | +0.008 | 0.966 |
| B3: Mode-Aware Target | 0.489 | 0.749 | 10.75 | 1.313 | -0.046 | 0.968 |
| B4: LinReg+Thermal | 0.488 | 0.760 | 10.70 | 1.239 | +0.002 | 0.967 |
| B5: LinReg+Mode | 0.496 | 0.738 | 10.70 | 1.389 | +0.004 | 0.969 |
| B6: LinReg+Gap | 0.496 | 0.738 | 10.70 | 1.390 | +0.004 | 0.969 |
| B7: LinReg+Time | 0.502 | 0.734 | 10.77 | 1.364 | +0.005 | 0.969 |
| B8: LinReg Full | 0.502 | 0.734 | 10.77 | 1.364 | +0.005 | 0.969 |
| B9: Per-Mode LinReg | 0.504 | 0.728 | 10.60 | 1.359 | +0.010 | 0.970 |

**Winner: Persistence (B1)** with 0.434°F MAE

---

## Results by Mode

### Heating Mode (19.8% of samples)

HVAC is warming the home toward the heat setpoint.

| Model | MAE | RMSE | P95 | Bias |
|-------|-----|------|-----|------|
| B1: Persistence | **0.481** | 0.860 | 2.000 | **-0.359** |
| B2: Thermal Drift | 0.509 | 0.871 | 1.995 | -0.378 |
| B3: Mode-Aware Target | 0.548 | 0.823 | 1.781 | -0.176 |
| B4: LinReg+Thermal | 0.540 | 0.843 | 1.818 | -0.304 |
| B5: LinReg+Mode | 0.592 | 0.791 | 1.573 | +0.000 |
| B9: Per-Mode LinReg | 0.586 | 0.773 | 1.559 | -0.000 |

**Key insight:** Persistence has -0.36°F bias (temperature rises but we predict no change). Linear models correct the bias to ~0 but MAE gets worse.

### Cooling Mode (7.5% of samples)

HVAC is cooling the home toward the cool setpoint.

| Model | MAE | RMSE | P95 | Bias |
|-------|-----|------|-----|------|
| B1: Persistence | **0.522** | 0.902 | 2.000 | **+0.291** |
| B2: Thermal Drift | 0.547 | 0.912 | 2.030 | +0.319 |
| B4: LinReg+Thermal | 0.590 | 0.879 | 1.888 | +0.163 |
| B9: Per-Mode LinReg | 0.610 | 0.835 | 1.583 | -0.050 |

**Key insight:** Persistence has +0.29°F bias (temperature falls but we predict no change). Again, correcting bias doesn't improve MAE.

### Passive Mode (72.7% of samples)

HVAC is off, temperature drifting toward outdoor.

| Model | MAE | RMSE | P95 | Bias |
|-------|-----|------|-----|------|
| B1: Persistence | **0.413** | 0.731 | 1.000 | +0.067 |
| B2: Thermal Drift | 0.427 | 0.725 | 1.063 | +0.082 |
| B5: LinReg+Mode | 0.452 | 0.708 | 1.148 | +0.016 |

**Key insight:** Persistence is nearly unbiased and wins decisively. Temperature barely changes when HVAC is off.

---

## Summary by Mode

| Model | Heating | Cooling | Passive | Overall |
|-------|---------|---------|---------|---------|
| B1: Persistence | **0.481** | **0.522** | **0.413** | **0.434** |
| B2: Thermal Drift | 0.509 | 0.547 | 0.427 | 0.452 |
| B3: Mode-Aware Target | 0.548 | 0.622 | 0.459 | 0.489 |
| B4: LinReg+Thermal | 0.540 | 0.590 | 0.463 | 0.488 |
| B9: Per-Mode LinReg | 0.586 | 0.610 | 0.471 | 0.504 |

---

## Key Findings

### 1. Persistence is unbeatable with linear models

Despite having systematic bias in heating/cooling modes, persistence wins on MAE everywhere. Adding features and complexity makes things worse.

### 2. The Bias-Variance Paradox

| Mode | Persistence Bias | Expected Change | Actual Result |
|------|------------------|-----------------|---------------|
| Heating | -0.36°F | Temp rises ~0.36°F | Models that predict rise have higher MAE |
| Cooling | +0.29°F | Temp falls ~0.29°F | Models that predict fall have higher MAE |
| Passive | +0.07°F | Temp stable | Persistence nearly optimal |

**Why?** MAE is minimized at the **median**, not the mean. The temperature change distribution is likely:
- Centered near 0 (most of the time, temp barely changes)
- Heavy-tailed (occasionally, temp changes a lot)

Predicting the median (≈0, i.e., persistence) beats predicting the mean.

### 3. Passive mode dominates (73%)

Most of the time, HVAC maintains temperature in the deadband. In passive mode:
- Temperature barely changes
- Persistence is nearly optimal
- This pulls overall metrics toward persistence

### 4. 30 minutes may be too short

At this horizon:
- Temperature changes are small (median ≈ 0)
- Signal-to-noise ratio is low
- Persistence exploits the "no change" prior effectively

---

## Implications for RNN

For the RNN to beat persistence, it must:

1. **Capture non-linear dynamics** - Linear models can't exploit the signal
2. **Learn home-specific characteristics** - Thermal mass, insulation, HVAC power vary by home
3. **Use longer context** - Recent temperature trajectory might reveal patterns
4. **Focus on active modes** - Heating/cooling have more predictable dynamics

### Target Metrics

| Mode | Persistence MAE | Target for RNN |
|------|-----------------|----------------|
| Heating | 0.481 | < 0.45 |
| Cooling | 0.522 | < 0.48 |
| Passive | 0.413 | < 0.40 |
| **Overall** | **0.434** | **< 0.42** |

Even a 5% improvement over persistence would be significant given how strong the baseline is.

---

## Model Descriptions

See [BASELINE_MODELS.md](./BASELINE_MODELS.md) for detailed descriptions of each baseline model.
