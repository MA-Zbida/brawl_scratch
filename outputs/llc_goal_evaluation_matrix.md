# LLC Goal Evaluation Matrix

This matrix maps the current 11-feature curriculum goal space to the game skills you are training, the logs that prove progress, and the plots to inspect after each phase.

## Shared Retention Gate

For each phase `i`, the evaluator computes:

```text
retention_i(t) = current_score_i(t) / best_score_i_so_far
amnesia_i(t) = max(0, 1 - retention_i(t))
```

Advance only when the current phase reaches its minimum skill score and all previously trained phases have `retention >= 0.85` and `amnesia <= 0.15`.

| Phase | Minimum Skill Score |
|---|---:|
| `recovery_mastery` | 0.65 |
| `movement_fluency` | 0.65 |
| `weapon_acquisition` | 0.60 |
| `spacing_neutral` | 0.55 |
| `combat_execution` | 0.50 |
| `all_skills_llc` | 0.60 |

Retention evidence comes from `train/models/llc_<phase>_retention_eval.csv`, `train/models/llc_retention_best.json`, and `train/models/llc_<phase>_retention_amnesia.png`.

During training, `python tools/llc_live_monitor.py --phase <phase>` reads the same step, episode, and eval CSVs and prints live `STOP SIGNALS` for idle collapse, low action entropy, whiff spam, negative combat trade, and retention/amnesia failure.

After plots and the phase report, record your visual judgment with `python tools/record_llc_observation.py --phase <phase> --approved yes|no --notes "..."`. The next-action advisor treats this as a required approval gate.

## Feature Space

| Index | Feature | Skill Family |
|---:|---|---|
| 0 | `signed_dx_to_ledge` | Recovery |
| 1 | `dy_to_ledge` | Recovery |
| 2 | `player_x` | Movement |
| 3 | `player_y` | Movement |
| 4 | `player_has_weapon` | Weapon |
| 5 | `weapon_dx` | Weapon |
| 6 | `weapon_dy` | Weapon |
| 7 | `rel_distance` | Spacing |
| 8 | `rel_dy` | Spacing |
| 9 | `in_strike_range` | Combat |
| 10 | `frame_advantage_estimate` | Combat |

Step CSV evidence is written to `train/models/llc_<phase>_steps.csv`. Important shared columns are `goal_type`, `active_feature_errors`, `raw_goal_feats`, `goal_target`, `goal_mask`, `goal_error`, `goal_progress`, `goal_success`, `movement`, `jump`, `dodge`, `attack`, `idle`, `hit`, `whiff`, `death_event`, `damage_trade`, and combat reward components.

Episode CSV evidence is written to `train/models/llc_<phase>_episodes.csv`. Important shared columns are `episode_success`, `mean_goal_error`, `time_to_success`, `damage_trade`, `action_entropy`, `idle_rate`, `whiff_rate`, and `attack_precision`.

## Per-Goal Evaluation

| Goal | Active Features | Primary Metrics | Collapse / Amnesia Signals | Required Plots |
|---|---|---|---|---|
| `recovery_mastery` | `signed_dx_to_ledge`, `dy_to_ledge` | `episode_success_rate`, `mean_goal_error`, `death_event`, `time_to_success`, jump usage through action logs | rising death rate, ledge error stops decreasing, later phases drop recovery retention below 0.85 | `goal_family_errors`, `goal_feature_traces`, `goal_phase_spaces`, `retention_amnesia` |
| `movement_fluency` | `player_x`, `player_y` | target success, `mean_goal_error`, `time_to_success`, `idle_rate`, `action_entropy` | idle rate above 0.45, action entropy collapses, movement retention below 0.85 | `goal_feature_traces`, `goal_phase_spaces`, `episode_health`, `retention_amnesia` |
| `weapon_acquisition` | `player_has_weapon`, `weapon_dx`, `weapon_dy` | `weapon_pickup_rate`, pickup success, time-to-pickup through `time_to_success`, weapon error | pickup rate falls after combat, `weapon_drop_event` rises, weapon retention below 0.85 | `goal_family_errors`, `goal_feature_traces`, `goal_phase_spaces`, `episode_health` |
| `spacing_neutral` | `rel_distance`, `rel_dy` | desired-band occupancy from active errors, vertical alignment, self damage, approach/retreat balance through action channels | spacing success falls when combat begins, self damage rises, spacing retention below 0.85 | `goal_family_errors`, `goal_feature_traces`, `goal_phase_spaces`, `episode_health` |
| `combat_execution` | `in_strike_range`, `frame_advantage_estimate` | `hit_rate`, `attack_precision`, `whiff_rate`, `mean_op_damage`, `mean_self_damage`, `mean_damage_trade`, `win_rate` | attack spam with high whiff, negative damage trade, movement/spacing retention collapse | `combat_precision`, `episode_health`, `goal_feature_traces`, `retention_amnesia` |
| `all_skills_llc` | sampled from all five families | per-family retention, aggregate skill score, positive final damage trade | any previous phase `amnesia > 0.15`, one goal family dominates logs, combat improves while movement/recovery collapse | all six diagnostics: `retention_amnesia`, `goal_family_errors`, `goal_feature_traces`, `goal_phase_spaces`, `episode_health`, `combat_precision` |

## Manual Observation Checklist

Use this checklist next to the generated `outputs/llc_<phase>_run_report.md` after each phase:

| Question | Stop If The Answer Is Yes |
|---|---|
| Does movement visibly collapse into idling or jitter? | Recheck perception, refresh movement demos, rehearse with higher entropy. |
| Does recovery still fail from the same offstage starts? | Collect recovery demos with sequence enforcement and lower ledge error before advancing. |
| Does weapon pickup look accidental instead of intentional? | Recollect weapon demos and inspect `weapon_dx/weapon_dy` traces. |
| Does combat spam attacks outside strike range? | Rehearse spacing first, then combat with whiff gate enforced. |
| Does current combat improve while old phase retention falls? | Stop ladder training and run a 100k rehearsal with all demos so far. |
