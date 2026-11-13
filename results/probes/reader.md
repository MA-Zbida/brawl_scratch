# The recurrent readers: what each fighter is doing, from its crop over time

2026-11-13 15:14. scripts/train_reader.py. Labels from key presses (brawl/reader.py). AUC: 0.5 is chance. A reader never saw the human sessions it is tested on; their crops come from recorded JPEG frames, the scripted ones from the live screen.

## Wu Shang

Trained on 60 min of scripted play (20261113_121255); tested on 110 min of human play (9 sessions), where the fighter is attacking in 39.7% of decisions, dodging in 13.5%, jumping in 7.5%.

| Input | Play | Attacking (AUC) | Dodging (AUC) | Jumping (AUC) | Since the last dodge (error, decisions) | Jumps since the ground (error) |
|---|---|---|---|---|---|---|
| kinematics | scripted, held back | 0.697 | 0.699 | 0.907 | 23.1 | 0.23 |
| kinematics | human | 0.553 | 0.698 | 0.795 | 26.6 | 0.54 |
| crop + kinematics | scripted, held back | 0.840 | 0.809 | 0.915 | 17.4 | 0.23 |
| crop + kinematics | human | 0.762 | 0.741 | 0.854 | 21.8 | 0.51 |

## Mordex

Trained on 60 min of scripted play (20261113_110925); tested on 14 min of human play (2 sessions), where the fighter is attacking in 35.3% of decisions, dodging in 10.8%, jumping in 7.4%.

| Input | Play | Attacking (AUC) | Dodging (AUC) | Jumping (AUC) | Since the last dodge (error, decisions) | Jumps since the ground (error) |
|---|---|---|---|---|---|---|
| kinematics | scripted, held back | 0.686 | 0.723 | 0.933 | 22.6 | 0.25 |
| kinematics | human | 0.620 | 0.612 | 0.770 | 28.6 | 0.52 |
| crop + kinematics | scripted, held back | 0.842 | 0.786 | 0.920 | 19.1 | 0.27 |
| crop + kinematics | human | 0.779 | 0.744 | 0.839 | 22.2 | 0.50 |

Outputs and hidden states written for 9 human Wu Shang sessions (reader_k3.npz).
