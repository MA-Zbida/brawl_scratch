# Probe: does a crop show what its fighter is doing?

2026-11-12 22:46. 9 medium sessions, held out whole (3 folds); AUC, mean ± spread over the folds (0.5 is chance). Labels from the human's key presses: Wu Shang is attacking in 16.5% of decisions, dodging in 9.1% of decisions, jumping in 5.7% of decisions. scripts/probe_state.py.

| Input (Wu Shang only) | Attacking | Dodging | Jumping |
|---|---|---|---|
| kinematics | 0.707 ± 0.007 | 0.812 ± 0.006 | 0.836 ± 0.013 |
| crop | 0.875 ± 0.002 | 0.891 ± 0.000 | 0.929 ± 0.004 |
| crop + kinematics | 0.886 ± 0.002 | 0.901 ± 0.002 | 0.932 ± 0.003 |

The crop model, trained on Wu Shang only, applied to Mordex's crop: how well each output ranks the decisions after which Wu Shang takes damage within 0.5 s (AUC; held-out sessions):

| Output for Mordex | All decisions | Fighters within 200 px |
|---|---|---|
| attacking | 0.566 | 0.551 |
| dodging | 0.538 | 0.526 |
| jumping | 0.494 | 0.486 |
| (for scale: being close, by the vector's distance) | 0.710 | |

The vector's own prediction of that damage reaches 0.76 (results/probes/vision.md).
