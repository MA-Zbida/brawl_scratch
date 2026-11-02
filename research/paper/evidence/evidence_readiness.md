# Paper evidence readiness

Only current canonical filenames are used. Historical 3h runs, pilots, failed runs, and alternate versions are never selected.

## What can be reported now

- Current demonstration composition and integrity measurements.
- Within-run learning diagnostics against cumulative live-game transitions.
- Preliminary checkpoint skill scores and retention with sample counts in every cell.

## What cannot yet be claimed

- Causal improvement from transfer, FiLM, replay, KL anchoring, or PCGrad.
- Optimization uncertainty across independent training seeds.
- Final goal-independent match competence.

Evaluation readiness requires at least 30 episodes per checkpoint/skill cell.

## Blocking evidence gaps

- movement_fluency retention evaluation uses only 2 evaluation episodes; minimum is 30
- weapon_acquisition retention evaluation is missing
- spacing_neutral retention evaluation uses only 3 evaluation episodes; minimum is 30
- combat_execution retention evaluation uses only 3 evaluation episodes; minimum is 30
- missing evaluation cell: weapon_acquisition checkpoint on movement_fluency
- missing evaluation cell: weapon_acquisition checkpoint on weapon_acquisition
- independent training seeds are not represented in this evidence bundle
- matched-budget scratch baselines are missing for transfer claims
- mechanism ablations are missing for FiLM/replay/KL/PCGrad claims
- goal-independent match-play benchmark is missing

## Current training logs

| Phase | Transitions | Episodes | Smoothing |
|---|---:|---:|---:|
| Movement | 4,047 | 55 | 20 episodes |
| Weapon | 200,172 | 1,716 | 20 episodes |
| Spacing | 50,105 | 1,069 | 20 episodes |
| Combat | 10,055 | 22 | 20 episodes |

## Canonical demonstrations

| Phase | Transitions | Episodes | Actions | Invalid goal rows |
|---|---:|---:|---:|---:|
| Movement | 1,596 | 50 | 11/27 | 0 |
| Weapon | 2,817 | 50 | 6/27 | 0 |
| Spacing | 1,970 | 50 | 9/27 | 0 |
| Combat | 13,205 | 50 | 11/27 | 0 |
