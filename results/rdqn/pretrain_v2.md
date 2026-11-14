# Pretrained recurrent Q-network: v2

2026-11-14 10:17. 15 human sessions (easy, medium, hard, extreme, chosen), 20000 updates of batch 16 sequences, margin 0.5, n-step 20, gamma 0.99, target every 500, learning rate 3e-05 (configs\experiments\rdqn_medium_v2.yaml). scripts/pretrain_rdqn.py.

Held-out check: trained on 10 sessions, played back greedily on 5 others (20261106_151844, 20261110_194554, 20261111_183313, 20261111_200054, 20261112_103941), memory carried through each session.

| Measure | Value |
|---|---|
| Action right (direction and button) | 0.419 |
| Baseline: repeat the human's previous action | 0.672 |
| Right where the human changes action | 0.232 |
| Human's new button presses matched | 0.146 |
| Decisions with a button: human / network | 0.300 / 0.214 |
| Different actions used by the network (of 54) | 52 |
| Share of its most used action | 0.460 |
| Repeats an action history taken from elsewhere (copying itself; 2026-11-13 network: 0.88-0.90) | 0.171 |

For scale, behaviour cloning (medium_v4) is right at 0.224 where the human changes action, on held-out stretches.

Trained on all 15 sessions -> runs\rdqn\pretrain_v2\qnet.pt
