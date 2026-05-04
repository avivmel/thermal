# MPC Demand-Response Evaluation Report

Status: paper-ready as a feasibility and pilot result, but not yet as a final statistically independent building-level experiment.

## Summary

We evaluated a first-passage MPC thermostat controller for residential HVAC demand response in 7-day ThermalGym episodes. The controller uses learned time-to-target predictions to precondition before a 17:00-20:00 peak window and then coast during the peak while preserving occupant comfort.

Across the screened evaluation cells, MPC reduced peak HVAC energy by 49.8% in cooling and 45.5% in heating at the median, with near-zero additional mode-aware comfort violation. Rule-based preconditioning and setback baselines reduced peak load more aggressively, but did so by violating comfort for long periods.

We also tested comfort-bounded versions of the rule baselines. These bounded rules clamp setpoints to the active comfort edge during the peak. They remove most of the rule-induced comfort penalty, but their median peak-energy reductions fall below MPC: 42.2-43.2% in cooling and 27.4-39.3% in heating. Against the best bounded rule in each mode, MPC improves median peak-energy reduction by 6.6 percentage points in cooling and 6.3 percentage points in heating.

The main paper claim should therefore be framed as:

> A first-passage MPC controller can shift roughly 45-50% of peak HVAC energy while maintaining comfort. Unbounded rule-based shedding reaches larger reductions by creating substantial comfort violations, while comfort-bounded rule baselines preserve comfort but reduce less peak energy than MPC at the median by about 6 percentage points.

The current result set contains repeated archetype labels. After deduplication, the defensible sample size is 3 unique cooling metric groups and 6 unique heating metric groups per controller, not 9 cooling cells and 18 heating cells.

## Evaluation Design

The evaluation uses 7-day ThermalGym episodes with:

| Setting | Value |
| --- | --- |
| Control step | 15 minutes |
| Peak window | 17:00-20:00 |
| Cooling comfort band | 72-76 deg F |
| Heating comfort band | 68-72 deg F |
| Forecast source | EPW weather |
| Evaluation output | `results/mpc_eval/selected_grid_expanded_safe` |

The baseline screen evaluated 54 candidate cells and retained 27 labeled cells:

| Mode | Labeled cells kept | Unique metric groups | Selected conditions |
| --- | ---: | ---: | --- |
| Cooling | 9 | 3 | Hot archetypes on 2017-06-15, 2017-07-15, 2017-08-15 |
| Heating | 18 | 6 | Cold and mixed archetypes on 2017-01-15, 2017-02-15, 2017-12-15 |

Deduplication groups labeled cells that share the same scenario metadata and rounded outcome metrics. This exposes repeated small/medium/large labels that produce identical controller outcomes.

The screening step rejected cells that either had insufficient baseline peak HVAC activity or severe baseline comfort-band problems:

| Mode | Kept | Rejected | Main rejection reasons |
| --- | ---: | ---: | --- |
| Cooling | 9 | 18 | Cold/mixed archetypes had weak or zero peak cooling load and large below-band violations |
| Heating | 18 | 9 | Hot archetypes had weak or unsuitable peak heating behavior and large above-band violations |

For the selected cells, the baseline peak-load ranges were:

| Mode | Baseline peak energy range | Baseline peak runtime range | Baseline band-violation fraction |
| --- | ---: | ---: | ---: |
| Cooling | 12.66-20.53 kWh | 1080-1125 min | 0.055-0.070 |
| Heating | 1.56-3.33 kWh | 1005-1260 min | 0.000 |

## Headline Results

Table 1 reports medians over labeled cells, with the deduplicated metric-group count shown explicitly. Because each repeated group appears three times, the medians match the deduplicated medians.

| Mode | Controller | Labeled cells | Unique metric groups | Peak energy reduction | Peak runtime reduction | Total energy overhead | Additional mode-aware comfort violation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cooling | MPC | 9 | 3 | 49.8% | 41.7% | +0.27% | 0 min |
| Cooling | Bounded precool | 9 | 3 | 42.2% | 43.1% | +1.98% | 0 min |
| Cooling | Bounded setback | 9 | 3 | 43.2% | 22.2% | -3.32% | 0 min |
| Cooling | Precool | 9 | 3 | 59.7% | 66.7% | +0.34% | 585 min |
| Cooling | Setback | 9 | 3 | 79.6% | 83.3% | -5.67% | 945 min |
| Heating | MPC | 18 | 6 | 45.5% | 42.6% | +0.58% | 0 min |
| Heating | Bounded preheat | 18 | 6 | 39.3% | 38.4% | +0.80% | 0 min |
| Heating | Bounded setback | 18 | 6 | 27.4% | 20.8% | -0.67% | 0 min |
| Heating | Preheat | 18 | 6 | 58.0% | 64.1% | +0.26% | 915 min |
| Heating | Setback | 18 | 6 | 79.6% | 80.8% | -2.07% | 1268 min |

MPC is the only non-baseline controller that preserves comfort in the median case for both heating and cooling. The December cold-heating group has a small MPC violation of 75 minutes, so heating MPC has a maximum group-level comfort violation of 75 minutes while the median remains 0 minutes.

The deduplicated MPC scenario outcomes are:

| Scenario | Date | Peak energy reduction | Peak runtime reduction | Total energy overhead | Added comfort violation | Peak energy | Peak runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hot cooling | 2017-06-15 | 57.1% | 56.9% | +1.54% | 0 min | 5.43 kWh | 465 min |
| Hot cooling | 2017-07-15 | 49.8% | 41.7% | +0.27% | 0 min | 9.12 kWh | 630 min |
| Hot cooling | 2017-08-15 | 44.4% | 25.3% | -0.38% | 0 min | 11.41 kWh | 840 min |
| Cold heating | 2017-01-15 | 44.5% | 31.0% | +0.50% | 0 min | 1.29 kWh | 870 min |
| Cold heating | 2017-02-15 | 34.5% | 31.0% | +0.65% | 0 min | 1.62 kWh | 870 min |
| Cold heating | 2017-12-15 | 30.7% | 21.4% | +0.15% | 75 min | 2.31 kWh | 990 min |
| Mixed heating | 2017-01-15 | 46.5% | 54.2% | +0.53% | 0 min | 1.10 kWh | 570 min |
| Mixed heating | 2017-02-15 | 55.4% | 62.7% | +0.79% | 0 min | 0.70 kWh | 375 min |
| Mixed heating | 2017-12-15 | 50.5% | 65.5% | +0.63% | 0 min | 0.94 kWh | 435 min |

## Interpretation

The MPC result is useful because it occupies a different operating point than the rule baselines:

| Controller family | Peak reduction | Comfort behavior | Interpretation |
| --- | --- | --- | --- |
| MPC | Moderate-high, about 45-50% median peak energy reduction | Median additional mode-aware comfort violation is 0 minutes | Comfort-constrained load shifting |
| Bounded preheat/precool | Slightly lower than MPC at the median | Median additional mode-aware comfort violation is 0 minutes | Comfort-bounded schedule without prediction |
| Bounded setback | Lower than MPC at the median | Median additional mode-aware comfort violation is 0 minutes | Comfort-edge peak setpoint relaxation |
| Preheat/precool | Higher than MPC | Hundreds of additional violation minutes | Over-aggressive thermal storage |
| Setback | Highest peak reduction | Largest comfort violations | Peak shedding by discomfort |

This matters for the paper because the rule baselines are not failed baselines. They demonstrate the expected tradeoff: it is easy to reduce peak HVAC energy by relaxing comfort, but harder to reduce peak energy while respecting comfort. MPC is favorable on that constrained objective.

## Rule-Based Baselines

The rule-based controllers in this evaluation are fixed setpoint-schedule baselines. They do not use the learned first-passage predictor, do not estimate whether the building can coast to the end of the peak window, and do not adapt preconditioning start time to the current indoor temperature or outdoor forecast.

The rules are intentionally simple. With the defaults used in this evaluation, the exact setpoints are:

| Mode | Controller | 15:00-17:00 | 17:00-20:00 | Comfort-bounded during peak? |
| --- | --- | ---: | ---: | --- |
| Cooling | Baseline | cool `75 F` | cool `75 F` | yes |
| Cooling | Precool | cool `73 F` | cool `77 F` | no, above `76 F` limit |
| Cooling | Bounded precool | cool `73 F` | cool `76 F` | yes |
| Cooling | Setback | cool `75 F` | cool `79 F` | no, above `76 F` limit |
| Cooling | Bounded setback | cool `75 F` | cool `76 F` | yes |
| Heating | Baseline | heat `69 F` | heat `69 F` | yes |
| Heating | Preheat | heat `71 F` | heat `67 F` | no, below `68 F` limit |
| Heating | Bounded preheat | heat `71 F` | heat `68 F` | yes |
| Heating | Setback | heat `69 F` | heat `65 F` | no, below `68 F` limit |
| Heating | Bounded setback | heat `69 F` | heat `68 F` | yes |

The mechanism is:

| Rule | Before peak | During peak | Intended mechanism |
| --- | --- | --- | --- |
| Precool | Lower cooling setpoint for two hours before a summer peak | Raise cooling setpoint during peak | Store cooling before peak, then reduce compressor calls during peak |
| Preheat | Raise heating setpoint for two hours before a winter peak | Lower heating setpoint during peak | Store heat before peak, then reduce heating calls during peak |
| Setback | No preconditioning | Relax the active comfort-side setpoint during peak | Shed peak runtime by allowing indoor temperature to drift farther |
| Bounded variants | Same timing as the corresponding rule | Clamp peak setpoint to the comfort edge | Test comfort-bounded scheduling without prediction |

These baselines answer a useful but limited question: how much peak reduction is available from simple setpoint manipulation if comfort is not tightly enforced? They do not answer the stronger question of how MPC compares against the best comfort-bounded rule controller.

Could the rule-based approach be bounded to comfort bands? Yes. The bounded variants above test exactly that. They are a fairer comparator than the original unbounded rules, although setpoint bounds still do not mathematically guarantee zero comfort violation because indoor temperature can drift or overshoot between control steps.

The bounded-rule results are:

| Mode | Controller | Unique groups | Peak energy reduction median | Peak runtime reduction median | Energy overhead median | Comfort violation median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cooling | MPC | 3 | 49.8% | 41.7% | +0.27% | 0 min |
| Cooling | Bounded precool | 3 | 42.2% | 43.1% | +1.98% | 0 min |
| Cooling | Bounded setback | 3 | 43.2% | 22.2% | -3.32% | 0 min |
| Heating | MPC | 6 | 45.5% | 42.6% | +0.58% | 0 min |
| Heating | Bounded preheat | 6 | 39.3% | 38.4% | +0.80% | 0 min |
| Heating | Bounded setback | 6 | 27.4% | 20.8% | -0.67% | 0 min |

The median advantage of MPC over the best comfort-bounded rule is:

| Mode | MPC peak-energy reduction | Best bounded rule | Best bounded peak-energy reduction | MPC advantage |
| --- | ---: | --- | ---: | ---: |
| Cooling | 49.8% | Bounded setback | 43.2% | +6.6 percentage points |
| Heating | 45.5% | Bounded preheat | 39.3% | +6.3 percentage points |

In relative terms, MPC gives about 15% more peak-energy reduction than the best bounded cooling rule and about 16% more than the best bounded heating rule. The bounded rules are not dominated in every individual scenario. In one cold-heating scenario, bounded preheat slightly exceeds MPC peak-energy reduction while preserving comfort. At the median level, however, MPC remains stronger in both cooling and heating.

The spread over deduplicated metric groups shows the same tradeoff:

| Mode | Controller | Unique groups | Peak energy reduction median | Peak energy reduction range | Added comfort violation median | Added comfort violation range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cooling | MPC | 3 | 49.8% | 44.4-57.1% | 0 min | 0-0 min |
| Cooling | Bounded precool | 3 | 42.2% | 39.3-47.8% | 0 min | 0-0 min |
| Cooling | Bounded setback | 3 | 43.2% | 38.4-53.3% | 0 min | 0-15 min |
| Cooling | Precool | 3 | 59.7% | 54.9-59.9% | 585 min | 330-855 min |
| Cooling | Setback | 3 | 79.6% | 79.1-79.9% | 945 min | 765-1080 min |
| Heating | MPC | 6 | 45.5% | 30.7-55.4% | 0 min | 0-75 min |
| Heating | Bounded preheat | 6 | 39.3% | 28.7-47.3% | 0 min | 0-90 min |
| Heating | Bounded setback | 6 | 27.4% | 19.4-35.0% | 0 min | 0-0 min |
| Heating | Preheat | 6 | 58.0% | 44.8-65.6% | 915 min | 780-1080 min |
| Heating | Setback | 6 | 79.6% | 65.8-85.4% | 1268 min | 1125-1425 min |

In the comparison table, the best unconstrained rule was usually setback, but it always came with substantial additional comfort violation. The best comfort-feasible bounded rule was usually bounded precool or bounded preheat; it generally reduced less peak energy than MPC, with the one cold-heating exception noted above. Across deduplicated scenario groups, MPC beats the best bounded rule by +5.1 to +9.2 percentage points in cooling. In heating, the scenario-level margin ranges from -1.4 to +17.9 percentage points; the negative case is the cold February heating scenario where bounded preheat slightly outperforms MPC.

## Comfort Metrics

The report uses two comfort diagnostics:

| Metric | Meaning | How to use in paper |
| --- | --- | --- |
| Mode-aware comfort violation | Counts too-cold minutes in heating and too-hot minutes in cooling | Primary comfort metric |
| Full comfort-band violation | Counts either side of the comfort band | Diagnostic only |

Mode-aware comfort is the right headline metric because the controller objective is directional. In cooling, for example, the controller should avoid making the home too hot. Some selected hot cooling cells have below-band minutes under both baseline and MPC due to simulator initial conditions, so full-band violation should be reported separately as a diagnostic.

The MPC decision trace also supports the intended control mechanism. Across the deduplicated scenarios, MPC issued preconditioning commands before the peak and mostly coasted during the peak:

| Scenario | Date | Precondition decisions | Peak coast decisions | Peak maintain decisions |
| --- | --- | ---: | ---: | ---: |
| Hot cooling | 2017-06-15 | 30 | 82 | 2 |
| Hot cooling | 2017-07-15 | 36 | 79 | 5 |
| Hot cooling | 2017-08-15 | 35 | 74 | 10 |
| Cold heating | 2017-01-15 | 23 | 65 | 19 |
| Cold heating | 2017-02-15 | 27 | 72 | 12 |
| Cold heating | 2017-12-15 | 25 | 60 | 24 |
| Mixed heating | 2017-01-15 | 21 | 69 | 15 |
| Mixed heating | 2017-02-15 | 20 | 75 | 9 |
| Mixed heating | 2017-12-15 | 20 | 70 | 14 |

With a 15-minute control step, the 17:00-20:00 peak window contains 84 peak decisions over each 7-day episode. The phase counts show that MPC is not merely applying a fixed setback; it repeatedly decides whether to coast or maintain near the comfort boundary.

## Deduplication Finding

The expanded grid initially looked like 27 selected evaluation cells. Deduplication shows that the current ThermalGym archetype labels do not all produce independent outcomes:

| Mode and condition | Labeled cells | Unique weather-pattern outcomes |
| --- | ---: | ---: |
| Cooling hot archetypes | 9 | 3 |
| Heating cold archetypes | 9 | 3 |
| Heating mixed archetypes | 9 | 3 |
| Total | 27 | 9 |

Controller-level summaries therefore report 3 unique cooling metric groups and 6 unique heating metric groups. Any manuscript table should include both counts or use only the deduplicated count.

## Paper-Ready Claims

The current results support these claims:

1. MPC reduced median peak HVAC energy by 49.8% in cooling and 45.5% in heating while adding 0 minutes of median mode-aware comfort violation.
2. Rule-based preconditioning and setback controllers achieved larger peak reductions, but at the cost of hundreds to more than one thousand minutes of additional comfort violation over a 7-day episode.
3. The first-passage MPC controller provides a stronger median comfort-constrained demand-response operating point than the bounded rule baselines tested here, improving peak-energy reduction by 6.6 percentage points in cooling and 6.3 percentage points in heating relative to the best bounded rule in each mode.
4. Deduplication shows that the present grid should be interpreted as 9 unique scenario-pattern outcomes, not 27 independent building outcomes.

The current results do not yet support these stronger claims:

1. Generalization across many independent buildings.
2. Statistically powered superiority over all rule-based demand-response strategies.
3. Final confidence intervals over independent building samples.

## Suggested Manuscript Text

Use this as a draft results paragraph:

> In the screened ThermalGym evaluation, the proposed first-passage MPC controller reduced median peak HVAC energy by 49.8% in cooling and 45.5% in heating relative to normal thermostat operation. These reductions were achieved with 0 minutes of median additional mode-aware comfort violation and less than 1% median total-energy overhead. Unbounded rule-based precooling/preheating and setback controllers reduced peak energy more aggressively, reaching 58-80% median peak-energy reductions, but introduced substantial comfort violations ranging from 585 to 1268 additional minutes over the 7-day episodes. Comfort-bounded rule variants eliminated the median comfort penalty but produced lower median peak-energy reductions than MPC: 42.2-43.2% in cooling and 27.4-39.3% in heating. Relative to the best bounded rule in each mode, MPC improved median peak-energy reduction by 6.6 percentage points in cooling and 6.3 percentage points in heating. These results suggest that first-passage MPC provides a useful comfort-constrained demand-response operating point: it sacrifices some peak reduction relative to aggressive setpoint rules while outperforming the tested comfort-bounded schedules at the median.

Use this as a draft limitation paragraph:

> The current evaluation should be interpreted as a pilot rather than a statistically independent building-level study. Although the screened grid contains 27 labeled cells, deduplication by scenario metadata and rounded outcome metrics shows that these collapse to 9 unique scenario-pattern outcomes: 3 cooling patterns and 6 heating patterns per controller. We therefore report both labeled-cell counts and deduplicated metric-group counts, and treat bootstrap intervals over labeled cells as diagnostic rather than inferential.

## Reproduction

```bash
python scripts/run_mpc_baseline_grid.py \
  --workers 3 \
  --out-dir results/mpc_eval/baseline_grid_expanded_safe \
  --evaluation-out-dir results/mpc_eval/selected_grid_expanded_safe

python scripts/run_mpc_controller_grid.py \
  results/mpc_eval/baseline_grid_expanded_safe/selection/controller_grid.csv \
  --out-dir results/mpc_eval/selected_grid_expanded_safe \
  --workers 4

python scripts/analyze_mpc_eval.py results/mpc_eval/selected_grid_expanded_safe
```

Key outputs:

- `results/mpc_eval/selected_grid_expanded_safe/analysis/report.md`
- `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/main_results.csv`
- `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/metric_groups.csv`
- `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/mpc_vs_rule_comparison.csv`
- `results/mpc_eval/bounded_rules_comparison/analysis/tables/main_results.csv`
- `results/mpc_eval/bounded_rules_comparison/analysis/tables/metric_groups.csv`
- `results/mpc_eval/bounded_rules_comparison/analysis/tables/mpc_vs_rule_comparison.csv`

Figure-ready data:

- Screening funnel: `results/mpc_eval/baseline_grid_expanded_safe/selection/selected_cells.csv` and `results/mpc_eval/baseline_grid_expanded_safe/selection/rejected_cells.csv`
- Peak-energy versus comfort tradeoff scatter: `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/metric_groups.csv`
- Scenario-level controller bars: `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/metric_groups.csv`
- MPC phase-count plot: `results/mpc_eval/selected_grid_expanded_safe/analysis/tables/cell_controller_metrics.csv`

## Next Step Before Final Paper Tables

The next engineering step is to make the simulation cells genuinely independent. Specifically, verify whether ThermalGym is applying distinct envelope and HVAC parameters for the small, medium, and large archetype labels. If those labels remain behaviorally identical, the final paper should either report deduplicated scenario-pattern results or replace the grid with distinct building configurations before claiming building-level generalization.
