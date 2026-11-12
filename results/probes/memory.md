# Gate S1 probe: memory

2026-11-11 22:52. 6 medium sessions, held out two at a time (3 folds), 2 seeds per fold; mean ± spread over the 6 runs, then the mean difference from `vector` in the same fold and seed (with its spread). Lower is better except AUC. scripts/probe_memory.py.

| Input | Hit within 0.5 s: AUC | Hit log loss | Mordex in 0.25 s (px) | Mordex in 0.5 s (px) | Action NLL when the human changes | Action NLL |
|---|---|---|---|---|---|---|
| vector | 0.712 ± 0.021 | 0.1920 ± 0.0125 | 74.7 ± 0.4 | 133.5 ± 0.9 | 2.426 ± 0.080 | 1.026 ± 0.053 |
| vector + timers | 0.736 ± 0.026 (+0.024 ± 0.014) | 0.1874 ± 0.0120 (-0.0046 ± 0.0027) | 74.2 ± 0.6 (-0.5 ± 0.8) | 132.1 ± 1.0 (-1.4 ± 0.9) | 2.393 ± 0.076 (-0.033 ± 0.032) | 1.017 ± 0.049 (-0.009 ± 0.008) |
| vector, GRU | 0.707 ± 0.027 (-0.005 ± 0.014) | 0.1915 ± 0.0117 (-0.0005 ± 0.0012) | 73.8 ± 0.8 (-0.9 ± 1.0) | 131.8 ± 1.1 (-1.6 ± 0.9) | 2.355 ± 0.077 (-0.072 ± 0.046) | 0.995 ± 0.051 (-0.031 ± 0.012) |
| vector + timers, GRU | 0.725 ± 0.034 (+0.013 ± 0.020) | 0.1891 ± 0.0124 (-0.0029 ± 0.0022) | 73.4 ± 0.9 (-1.4 ± 0.8) | 130.6 ± 0.5 (-2.8 ± 0.9) | 2.313 ± 0.068 (-0.113 ± 0.040) | 0.989 ± 0.054 (-0.037 ± 0.007) |

Timers: since_press_jump, since_press_light, since_press_heavy, since_press_dodge, since_press_throw, agent_since_ground, opponent_since_ground, agent_jumps_since_ground, opponent_rises_since_ground, agent_since_hit, opponent_since_hit, agent_since_respawn, opponent_since_respawn (decisions, clipped at 80).
