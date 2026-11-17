# Distilled clone: advised_v1

2026-11-17 11:38. scripts/distill.py: behaviour cloning on the advised play of past nights (stretches kept: won). 426 stretches, 371715 decisions to train on, 69011 held out. Best epoch 23.

| Measure (held-out stretches) | Value |
|---|---|
| Loss | 0.887 |
| Both right | 0.726 |
| Baseline: repeat the previous action | 0.698 |
| Right where the action changes | 0.192 |
| Decisions with a button, played | 0.282 |
| Decisions with a button, most likely action | 0.233 |

Nights:

- rdqn_medium_v3\default\seed0: 236 stretches kept, 258192 of 429200 advised decisions (215 of 358 min)
- advised_medium_night2\default\seed0: 190 stretches kept, 182534 of 361200 advised decisions (152 of 301 min)
