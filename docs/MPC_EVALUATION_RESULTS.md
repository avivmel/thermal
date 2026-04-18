# MPC Evaluation Results

Status: expanded pilot results are reproducible and directionally useful for paper tables, but the labeled cell count still overstates independent building diversity.

## Evaluation Setup

The screened evaluation uses 7-day ThermalGym episodes with a 17:00-20:00 peak window and 15-minute control steps. A baseline-only sweep first runs all 9 archetype buildings across three winter heating windows and three summer cooling windows. The selector keeps cells with enough baseline peak load and acceptable baseline comfort-band behavior, then emits the controller grid used for the controller comparison.

Reproduction:

```bash
python scripts/run_mpc_baseline_grid.py \
  --workers 3 \
  --out-dir results/mpc_eval/baseline_grid_expanded_safe \
  --evaluation-out-dir results/mpc_eval/selected_grid_expanded_safe

python scripts/run_mpc_controller_grid.py \
  results/mpc_eval/baseline_grid_expanded_safe/selection/controller_grid.csv \
  --out-dir results/mpc_eval/selected_grid_expanded_safe \
  --workers 4
```

Compact outputs:

- `results/mpc_eval/baseline_grid_expanded_safe/selection/selected_cells.csv`
- `results/mpc_eval/baseline_grid_expanded_safe/selection/rejected_cells.csv`
- `results/mpc_eval/baseline_grid_expanded_safe/selection/controller_grid.csv`
- `results/mpc_eval/selected_grid_expanded_safe/analysis/report.md`
- `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/main_results.csv`

Raw per-run outputs are intentionally not needed for the paper-facing summary.

## Screened Cells

The expanded baseline screen evaluated 54 candidates and kept 27:

| Mode | Kept cells | Rejected cells | Rejection pattern |
| --- | ---: | ---: | --- |
| Cooling | 9 | 18 | Cold/mixed archetypes had zero or weak peak cooling load and large full-band cold violations |
| Heating | 18 | 9 | Hot archetypes had weak or unsuitable peak heating behavior and large full-band hot violations |

Selected cells:

| Mode | Buildings | Dates |
| --- | --- | --- |
| Cooling | `small_hot_heatpump`, `medium_hot_heatpump`, `large_hot_ac` | `2017-06-15`, `2017-07-15`, `2017-08-15` |
| Heating | `small_cold_heatpump`, `medium_cold_heatpump`, `large_cold_resistance` | `2017-01-15`, `2017-02-15`, `2017-12-15` |
| Heating | `small_mixed_heatpump`, `medium_mixed_heatpump`, `large_mixed_ac` | `2017-01-15`, `2017-02-15`, `2017-12-15` |

## Headline Results

The most useful current result is that MPC reduces peak energy and runtime with near-zero mode-aware comfort violations, while rule baselines shed more peak load by violating comfort for long periods.

| Mode | Controller | Cells | Peak energy reduction, median | Peak runtime reduction, median | Total energy overhead, median | Mode-aware comfort violation, median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cooling | MPC | 9 | 49.8% | 41.7% | +0.27% | 0 min |
| Cooling | Precool | 9 | 59.7% | 66.7% | +0.34% | 585 min |
| Cooling | Setback | 9 | 79.6% | 83.3% | -5.67% | 945 min |
| Heating | MPC | 18 | 45.5% | 42.6% | +0.58% | 0 min |
| Heating | Preheat | 18 | 58.0% | 64.1% | +0.26% | 915 min |
| Heating | Setback | 18 | 79.6% | 80.8% | -2.07% | 1268 min |

Interpretation: rule baselines can shed more peak load, but they do so by violating comfort for long periods. MPC gives a smaller peak reduction while mostly preserving the mode-aware comfort objective. The heating MPC mean comfort violation is 12.5 minutes because the December cold-archetype cases show a small 75-minute violation on each repeated building label; the median remains 0 minutes.

## Comfort-Band Diagnostics

The headline comfort metric is mode-aware:

- Heating counts too-cold minutes.
- Cooling counts too-hot minutes.

Full comfort-band diagnostics are still reported because they expose simulator initial-condition issues. In the selected hot cooling cells, both baseline and MPC have below-band minutes before or outside the cooling-control objective, with no incremental band violation from MPC. This is not a cooling-control failure, but it means the full-band diagnostic should be reported separately from the mode-aware comfort metric.

## Paper-Readiness Caveat

These results should not yet be presented as a statistically independent `n=27` experiment. The added dates create real weather variation, but the selected ThermalGym archetypes still repeat identical metrics within building-label groups:

- Cooling: 9 labeled cells collapse to 3 unique weather-pattern outcomes.
- Cold heating: 9 labeled cells collapse to 3 unique weather-pattern outcomes.
- Mixed heating: 9 labeled cells collapse to 3 unique weather-pattern outcomes.

The defensible interpretation today is closer to 9 unique scenario patterns, not 27 independent building outcomes. This is a stronger pilot than the original single-date screen, but final paper claims should either count unique scenario patterns explicitly or fix the scenario construction so the small/medium/large archetype labels produce distinct thermal/HVAC behavior.

## Next Step

Before freezing paper results, make the evaluation grid produce genuinely distinct building cells. The most direct options are:

1. Add a deduplication diagnostic to the analysis output so repeated metric groups are counted explicitly.
2. Validate whether ThermalGym building metadata is actually changing envelope/HVAC parameters across small/medium/large variants.
3. If the archetype variants are not distinct, report unique archetype-weather combinations or fix the scenario construction.

After that, rerun the selected controller grid and regenerate `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/main_results.csv` for the paper tables.
