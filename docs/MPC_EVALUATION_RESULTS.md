# MPC Evaluation Results

Status: pilot results are reproducible and useful for method development, but not yet strong enough for final paper claims.

## Evaluation Setup

The screened evaluation uses 7-day ThermalGym episodes with a 17:00-20:00 peak window and 15-minute control steps. A baseline-only sweep first runs all 9 archetype buildings under winter heating (`2017-01-15`) and summer cooling (`2017-07-15`). The selector keeps cells with enough baseline peak load and acceptable baseline comfort-band behavior, then emits the controller grid used for the comparison.

Reproduction:

```bash
python scripts/run_mpc_baseline_grid.py --workers 1
bash results/mpc_eval/baseline_grid/selection/run_commands.sh
```

Compact outputs:

- `results/mpc_eval/baseline_grid/selection/selected_cells.csv`
- `results/mpc_eval/baseline_grid/selection/rejected_cells.csv`
- `results/mpc_eval/baseline_grid/selection/controller_grid.csv`
- `results/mpc_eval/selected_grid/analysis/report.md`
- `results/mpc_eval/selected_grid/analysis/tables/main_results.csv`

Raw per-run outputs are intentionally not needed for the paper-facing summary.

## Screened Cells

The baseline screen evaluated 18 candidates and kept 9:

| Mode | Kept cells | Rejected cells | Rejection pattern |
| --- | ---: | ---: | --- |
| Cooling | 3 | 6 | Cold/mixed archetypes had zero baseline peak cooling load and large full-band cold violations |
| Heating | 6 | 3 | Hot archetypes had weak peak heating load and large full-band hot violations |

Selected cells:

| Mode | Buildings | Date |
| --- | --- | --- |
| Cooling | `small_hot_heatpump`, `medium_hot_heatpump`, `large_hot_ac` | `2017-07-15` |
| Heating | `small_cold_heatpump`, `medium_cold_heatpump`, `large_cold_resistance` | `2017-01-15` |
| Heating | `small_mixed_heatpump`, `medium_mixed_heatpump`, `large_mixed_ac` | `2017-01-15` |

## Headline Results

The most useful current result is that MPC reduces peak energy and runtime while adding no mode-aware comfort violations in the screened cells.

| Mode | Controller | Cells | Peak energy reduction, median | Peak runtime reduction, median | Total energy overhead, median | Mode-aware comfort violation, median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cooling | MPC | 3 | 49.8% | 41.7% | +0.27% | 0 min |
| Cooling | Precool | 3 | 59.7% | 66.7% | +0.34% | 585 min |
| Cooling | Setback | 3 | 79.6% | 83.3% | -6.80% | 945 min |
| Heating | MPC | 6 | 45.5% | 42.6% | +0.52% | 0 min |
| Heating | Preheat | 6 | 58.0% | 64.1% | +0.26% | 915 min |
| Heating | Setback | 6 | 79.6% | 80.8% | -2.12% | 1268 min |

Interpretation: rule baselines can shed more peak load, but they do so by violating comfort for long periods. MPC gives a smaller peak reduction while preserving the mode-aware comfort objective.

## Comfort-Band Diagnostics

The headline comfort metric is mode-aware:

- Heating counts too-cold minutes.
- Cooling counts too-hot minutes.

Full comfort-band diagnostics are still reported because they expose simulator initial-condition issues. In the selected hot cooling cells, both baseline and MPC have 660 minutes below the full 72-76 F cooling comfort band, with no incremental band violation from MPC. This is not a cooling-control failure, but it means the full-band diagnostic should be reported separately from the mode-aware comfort metric.

## Paper-Readiness Caveat

These results should not yet be presented as a statistically independent `n=9` experiment. The selected ThermalGym archetypes repeat identical metrics within climate/mode groups:

- Hot cooling: 3 buildings, identical selected-grid metrics.
- Cold heating: 3 buildings, identical selected-grid metrics.
- Mixed heating: 3 buildings, identical selected-grid metrics.

The defensible interpretation today is closer to 3 unique scenario patterns, not 9 independent building outcomes. This is sufficient for a reproducible pilot and ablation target, but not final paper evidence.

## Next Step

Before freezing paper results, make the evaluation grid produce genuinely distinct cells. The most direct options are:

1. Add multiple start dates per selected climate/mode so each archetype sees different weather and initial states.
2. Add a deduplication diagnostic to the analysis output so repeated metric groups are counted explicitly.
3. Validate whether ThermalGym building metadata is actually changing envelope/HVAC parameters across small/medium/large variants; if not, use only unique archetype-weather combinations or fix the scenario construction.

After that, rerun the selected controller grid and regenerate `results/mpc_eval/selected_grid/analysis/tables/main_results.csv` for the paper tables.
