# Pretrained recurrent Q-network: v3

2026-11-14 23:06. 15 human sessions (easy, medium, hard, extreme, chosen), 20000 updates of batch 16 sequences, margin 0.5, n-step 20, gamma 0.99, target every 500, learning rate 3e-05 (configs\experiments\rdqn_medium_v3.yaml). scripts/pretrain_rdqn.py.

Trained on all 15 sessions -> runs\rdqn\pretrain_v3\qnet.pt
